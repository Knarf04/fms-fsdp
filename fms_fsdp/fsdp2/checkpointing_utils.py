import os
import time

import torch
from torch.distributed.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    load_state_dict,
    save_state_dict,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)

# FSDP2 imports
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)

from fms_fsdp.utils.checkpointing_utils import Checkpointer


class Checkpointer_FSDP2(Checkpointer):
    """
    FSDP2-compatible checkpoint manager inheriting from the original Checkpointer.
    Overrides save/load methods to use torch.distributed.checkpoint (DCP) state_dict APIs
    instead of FSDP1 context managers.
    """

    def __init__(
        self,
        ckpdir,
        n_to_save,
        parallel_mode,
        rank,
        local_rank,
        report_fn=None,
        model_auto_placement=False, # Unused in FSDP2, kept for signature compatibility
    ):
        super().__init__(
            ckpdir, 
            n_to_save, 
            parallel_mode, 
            rank, 
            local_rank, 
            report_fn, 
            model_auto_placement
        )

    def _write(self, state_dict, loader_state, process_group, save_name, rank):
        os.makedirs(save_name, exist_ok=True)
        # dist.barrier()
        writer = FileSystemWriter(save_name, single_file_per_rank=True)
        if state_dict is not None:
            # FSDP2/DCP save is collective. We ignore 'process_group' arg here 
            # and let DCP use the default group unless you specifically need to restrict it.
            save_state_dict(
                state_dict=state_dict,
                storage_writer=writer,
                planner=DefaultSavePlanner(),
            )
        if loader_state is not None:
            loader_state.save_to_path(save_name)

    # --------------------------------------------------------------------------
    # Public API Overrides (FSDP2 Logic)
    # --------------------------------------------------------------------------

    def load(
        self,
        model,
        optimizer,
        dataloader,
        path="",
        reset_stepcount=False,
        strict=True,
        is_compiled=False,
    ):
        is_resuming = False
        if self._validate_ckp_path(self.ckp_path) is not None:
            path = self.ckp_path
            is_resuming = True
        load_path = self._validate_ckp_path(path)
        if load_path is None:
            self.report(
                f"No valid checkpoint detected at {path}, starting from scratch."
            )
            return model, optimizer, dataloader, 0, 0, False
        else:
            self.report(f"Prior checkpoint {load_path} detected.")
            model_load_time = time.time()
            if os.path.isfile(load_path):
                checkpoint_data = torch.load(load_path, map_location="cpu", weights_only=False)
                model_state = checkpoint_data.get("model_state")
                target_model = model._orig_mod if is_compiled else model
                options = StateDictOptions(
                    strict=strict,
                    full_state_dict=True
                )

                # FSDP2 API: Set full state dict
                set_model_state_dict(
                    target_model,
                    model_state,
                    options=options,
                )
                self.report(
                    f"Checkpoint {load_path} is a single-file checkpoint containing only a model. Optimizer and dataloader are from scratch.",
                    model_load_time=time.time() - model_load_time,
                )
                return model, optimizer, dataloader, 0, 0, is_resuming
            else:
                model_state = get_model_state_dict(model)
                model_ckp = {"model_state": model_state}
                load_state_dict(
                    state_dict=model_ckp,
                    storage_reader=FileSystemReader(load_path),
                    planner=DefaultLoadPlanner(),
                )
                
                set_model_state_dict(
                    model,
                    model_ckp["model_state"],
                    options=StateDictOptions(strict=strict),
                )

                self.report(model_load_time=time.time() - model_load_time)
                step = 0
                ntok = 0
                # Load metadata
                if is_resuming:
                    metadata = torch.load(os.path.join(load_path, "metadata.pth"), weights_only=False)
                    step = metadata.get("step", 0)
                    ntok = metadata.get("tokens_seen", 0)
                    self.report("Metadata loaded", start_step=step, n_tokens_seen=ntok)
                # Load optimizer
                if optimizer is not None:
                    optim_load_time = time.time()
                    optim_state = get_optimizer_state_dict(model, optimizer)
                    optim_ckp = {"optimizer_state": optim_state}
                    load_state_dict(
                        state_dict=optim_ckp,
                        storage_reader=FileSystemReader(load_path),
                        planner=DefaultLoadPlanner(),
                    )
                    
                    set_optimizer_state_dict(
                        model,
                        optimizer,
                        optim_state_dict=optim_ckp["optimizer_state"],
                    )
                    self.report(optimizer_load_time=time.time() - optim_load_time)
                else:
                    self.report("Skipping optimizer load, no optimizer provided.")
                # Load dataset
                if dataloader is not None:
                    data_load_time = time.time()
                    dataloader.dataset.load_from_path(path)
                    self.report(dataset_load_time=time.time() - data_load_time)
                else:
                    self.report("Skipping dataset load, no dataloader provided.")
                return model, optimizer, dataloader, step, ntok, is_resuming

    def save(
        self,
        step,
        model,
        optimizer,
        dataloader,
        **kwargs,
    ):
        rank = self.rank
        save_time = time.time()

        # FSDP2: Retrieve Sharded State Dicts (Lightweight metadata wrappers)
        model_state = get_model_state_dict(model)
        optim_state = get_optimizer_state_dict(model, optimizer)
        
        dataloader_state = None if dataloader is None else dataloader.dataset

        save_name = os.path.join(self.ckp_path, "step_" + str(step) + "_ckp")
        state_dict = {"model_state": model_state, "optimizer_state": optim_state}
        if self._do_save(rank, self.local_rank):
            self._write(
                state_dict, dataloader_state, None, save_name, rank
            )
        else:
            self._write(None, dataloader_state, None, save_name, rank)
        if rank == 0:
            metadata = kwargs
            metadata["step"] = step
            torch.save(metadata, os.path.join(save_name, "metadata.pth"))

        # dist.barrier()
        self.report(
            f"Checkpoint saved in {save_name}", model_save_time=time.time() - save_time
        )

        return self._cleanup()
    
    def save_single_file(
        self,
        step,
        model,
        is_compiled=False,
        **kwargs,
    ):
        pth_path = os.path.join(self.ckp_path[:-12], "pth", "step_" + str(step))
        os.makedirs(pth_path, exist_ok=True)
        save_name = os.path.join(pth_path, "consolidated.00.pth")
        save_time = time.time()

        # FSDP2: Full State Dict
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        target_model = model._orig_mod if is_compiled else model
        model_state = get_model_state_dict(target_model, options=options)

        if self.rank == 0:
            metadata = kwargs
            metadata["step"] = step
            metadata["model_state"] = model_state
            torch.save(metadata, save_name)
        # dist.barrier()
        self.report("Checkpoint written", model_save_time=time.time() - save_time)

        return self._cleanup()
