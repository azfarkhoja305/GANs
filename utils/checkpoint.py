from pathlib import Path
import torch.nn as nn
import numpy as np
import torch
import re
from copy import deepcopy
from utils.utils import load_params

Path.ls = lambda x: list(x.iterdir())


class Checkpoint:
    """Saves checkpoints at required epochs. Additionally
    automatically picks up the latest checkpoint if the folder already exists.
    Can also load the checkpoint given the file"""

    def __init__(self, ckp_folder, max_epochs, num_ckps, start_after=0.5):
        """Start checkpointing after `start_after*max_epoch`.
        Like start after 50% of max_epochs completed and divides the number of
        checkpoints equally."""
        self.ckp_folder = ckp_folder
        self.max_epochs = max_epochs
        self.num_ckps = num_ckps
        self.ckp_epochs = np.linspace(
            start_after * max_epochs, max_epochs, num_ckps, dtype=np.int
        ).tolist()
        if isinstance(self.ckp_folder, str):
            self.ckp_folder = Path(self.ckp_folder)

    def check_if_exists(self, generator, critic, gen_opt, critic_opt):
        if not self.ckp_folder.exists():
            self.ckp_folder.mkdir(parents=True)
            generator_avg_params = deepcopy(list(p.data for p in generator.parameters()))
            return generator, generator_avg_params, critic, gen_opt, critic_opt, 0, 0, None

        ckp_files = [
            file for file in self.ckp_folder.ls() if file.suffix in [".pth", ".pt"]
        ]
        if not ckp_files:
            generator_avg_params = deepcopy(list(p.data for p in generator.parameters()))
            return generator, generator_avg_params, critic, gen_opt, critic_opt, 0, 0, None
        print(
            "Checkpoint folder with checkpoints already exists. Searching for the latest."
        )
        # finding latest (NOT best) checkpoint to resume train
        numbers = [int(re.search(r"\d+", name.stem).group()) for name in ckp_files]
        idx = max(enumerate(numbers), key=lambda x: x[1])[0]
        return self.load_checkpoint(
            ckp_files[idx], generator, critic, gen_opt, critic_opt
        )

    def at_epoch_end(self, generator, generator_avg_params, critic, gen_opt, critic_opt, epoch, step, loss_logs):
        if epoch in self.ckp_epochs:
            self.save_checkpoint(
                self.ckp_folder / f"GanModel_{epoch:03}.pth",
                generator,
                critic,
                generator_avg_params,
                gen_opt,
                critic_opt,
                epoch,
                step,
                loss_logs,
            )

    @staticmethod
    def load_checkpoint(ckp_path, generator, critic, gen_opt=None, critic_opt=None):
        assert isinstance(generator, nn.Module), f"Generator is not nn.Module"
        assert isinstance(critic, nn.Module), f"Discriminator is not nn.Module"
        if isinstance(ckp_path, str):
            ckp_path = Path(ckp_path)
        assert ckp_path.exists(), f"Checkpoint File: {str(ckp_path)} does not exist"
        print(f"=> Loading checkpoint: {ckp_path}")
        ckp = torch.load(ckp_path)
        generator.load_state_dict(ckp["generator_state_dict"])
        critic.load_state_dict(ckp["critic_state_dict"])
        if gen_opt is not None and ckp["gen_optim_state_dict"] is not None:
            gen_opt.load_state_dict(ckp["gen_optim_state_dict"])
        if critic_opt is not None and ckp["critic_optim_state_dict"] is not None:
            critic_opt.load_state_dict(ckp["critic_optim_state_dict"])
        generator_avg_params = None
        if ckp["generator_avg_state_dict"] is not None:
            generator_avg = deepcopy(generator)
            generator_avg.load_state_dict(checkpoint['generator_avg_state_dict'])
            generator_avg_params = deepcopy(list(p.data for p in generator_avg.parameters()))


        epoch_complete = ckp["epoch"]
        loss_logs = ckp["loss_logs"]
        step = ckp["step"]
        return generator, generator_avg_params, critic, gen_opt, critic_opt, epoch_complete + 1, step, loss_logs

    @staticmethod
    def save_checkpoint(
        file_path,
        generator,
        critic,
        generator_avg_params=None,
        gen_opt=None,
        critic_opt=None,
        epoch=-1,
        step=-1,
        loss_logs=None,
    ):
        if isinstance(file_path, str):
            file_path = Path(file_path)
        assert (
            not file_path.is_dir()
        ), f"`file_path` cannot be a dir, Needs to be dir/file_name"
        ckp_suffix = [".pth", ".pt"]
        assert (
            file_path.suffix in ckp_suffix
        ), f"{file_path.name} is not in checkpoint file format"
        assert isinstance(generator, nn.Module), f"Generator is not nn.Module"
        generator_avg_state_dict = None
        if generator_avg_params:
            generator_avg = deepcopy(generator)
            load_params(generator_avg, generator_avg_params)
            generator_avg_state_dict = generator_avg.state_dict()
        assert isinstance(critic, nn.Module), f"Discriminator is not nn.Module"
        print(f"=> Saving Checkpoint with name `{file_path.name}`")
        gen_opt_dict = gen_opt.state_dict() if gen_opt is not None else None
        critic_opt_dict = critic_opt.state_dict() if critic_opt is not None else None
        torch.save(
            {
                "generator_state_dict": generator.state_dict(),
                "generator_avg_state_dict": generator_avg_state_dict,
                "critic_state_dict": critic.state_dict(),
                "gen_optim_state_dict": gen_opt_dict,
                "critic_optim_state_dict": critic_opt_dict,
                "epoch": epoch,
                "step": step,
                "loss_logs": loss_logs,
            },
            file_path,
        )

    @staticmethod
    def delete_checkpoint(file_path):
        if isinstance(file_path, str):
            file_path = Path(file_path)
        ckp_suffix = [".pth", ".pt"]
        assert (
            file_path.suffix in ckp_suffix
        ), f"{file_path.name} is not in checkpoint file format"
        assert file_path.exists(), f"`file_path`: {str(file_path)} not found"
        print(f"Deleting {str(file_path)}")
        file_path.unlink()

    def find_best_ckp(self):
        """ Calculate the metric for each checkpoint and return best"""
        raise NotImplementedError
