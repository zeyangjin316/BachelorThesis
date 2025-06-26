import os
import numpy as np
import pandas as pd


class ResultSaver:
    def __init__(self, mode, cgm_params, two_step_params):
        self.mode = mode
        self.cgm_params = cgm_params
        self.two_step_params = two_step_params
        self.base_path = self._build_base_path()
        os.makedirs(self.base_path, exist_ok=True)

    def _build_base_path(self):
        if self.mode == "cgm":
            name = (
                f"CGM_loss-{self.cgm_params['loss_type']}"
                f"_win{self.cgm_params['train_window_size']}"
                f"_tf{self.cgm_params['train_freq']}"
                f"_ep{self.cgm_params['n_epochs']}"
                f"_bs{self.cgm_params['batch_size']}"
                f"_ns{self.cgm_params['n_samples']}"
            )
        elif self.mode == "2step":
            name = (
                f"TWOSTEP_uv-{self.two_step_params['uv_method']}"
                f"_copula-{self.two_step_params['copula_type']}"
                f"_win{self.two_step_params['copula_window_size']}"
                f"_tf{self.two_step_params['uv_train_freq']}"
            )
        else:  # comparison
            name = (
                f"COMPARISON"
                f"_CGM-loss-{self.cgm_params['loss_type']}-win{self.cgm_params['train_window_size']}-samp{self.cgm_params['n_samples']}"
                f"_2STEP-{self.two_step_params['uv_method']}-{self.two_step_params['copula_type']}-w{self.two_step_params['copula_window_size']}"
            )
        return os.path.join("results", name)

    def save(self, samples_cgm, samples_two_step, df_summary):
        if samples_cgm is not None:
            np.save(os.path.join(self.base_path, "samples_cgm.npy"), samples_cgm)

        if samples_two_step is not None:
            np.save(os.path.join(self.base_path, "samples_two_step.npy"), samples_two_step)

        df_summary.to_csv(os.path.join(self.base_path, "summary.csv"), index=False)
        print(f"✅ Results saved to: {self.base_path}")
