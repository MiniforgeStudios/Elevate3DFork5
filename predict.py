import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from cog import BasePredictor, Input, Path as CogPath

# This predictor wraps the Elevate3D pipeline's main script.
# It expects to live at the root of the Elevate3D repo (next to main_refactor.py).

class Predictor(BasePredictor):
    def setup(self):
        # Validate that required files are present
        required = ["main_refactor.py", "requirements.txt"]
        for f in required:
            if not Path(f).exists():
                raise RuntimeError(f"Required file missing: {f}. Make sure this Predictor lives at the root of Elevate3D.")
        # Ensure PoissonRecon binary exists
        pr_bin = Path("PoissonRecon/Bin/Linux/PoissonRecon")
        if not pr_bin.exists():
            # Not fatal during setup—the build step should have compiled it,
            # but we'll warn so users know why geometry steps might fail.
            print("Warning: PoissonRecon executable not found at PoissonRecon/Bin/Linux/PoissonRecon. "
                  "Geometry refinement might fail unless built in the container build step.")

    def predict(
        self,
        model_file: CogPath = Input(description="Low-quality 3D model file (OBJ/GLB/PLY)."),
        prompt_text: Optional[str] = Input(default=None, description="Optional target appearance / guidance text."),
        obj_name: str = Input(default="asset", description="Object name (used for folder naming)."),
        dataset: str = Input(default="MyData", description="Dataset name to nest under Inputs/3D/."),
        conf_name: str = Input(default="config_my", description="Config name from Configs/ folder (default works to start)."),
        bake_mesh: bool = Input(default=True, description="Bake final mesh to file."),
        device_idx: int = Input(default=0, description="GPU device index inside container."),
        mode: str = Input(choices=["full", "texture_only", "geometry_only"], default="full", description="Run full pipeline or a subset."),
        return_zip: bool = Input(default=True, description="Return a ZIP of the Outputs folder."),
    ) -> CogPath:
        """Run Elevate3D pipeline on the uploaded model.

        This wraps the 2-stage flow from the README:
          1) Preprocess (normalize + render multi-views)
          2) Refinement via main_refactor.py

        The exact scripts may evolve upstream; adjust here accordingly.
        """
        model_path = Path(model_file)
        model_ext = model_path.suffix.lower()
        if model_ext not in [".obj", ".glb", ".ply"]:
            raise ValueError("Unsupported model format. Please upload .obj, .glb, or .ply")

        # Prepare inputs directory structure
        in_root = Path("Inputs/3D") / dataset / obj_name
        in_root.mkdir(parents=True, exist_ok=True)
        # Copy the uploaded file to expected input location
        target_model = in_root / f"{obj_name}{model_ext}"
        shutil.copyfile(model_path, target_model)

        # Write prompt if provided
        if prompt_text:
            (in_root / "prompt.txt").write_text(prompt_text)

        # Run preprocessing (if the project provides a helper script; otherwise, skip)
        # Fallback: call preprocess.py directly if available.
        preprocess_script = Path("run_preprocess_script.sh")
        if preprocess_script.exists():
            self._run_cmd(["bash", str(preprocess_script)], cwd=".")
        else:
            # Try a minimal preprocessing by calling preprocess.py if it exists.
            if Path("preprocess.py").exists():
                self._run_cmd(["python", "preprocess.py", "--obj_name", obj_name, "--dataset", dataset], cwd=".")
            else:
                print("Preprocess helpers not found. Proceeding directly to refinement...")

        # Build base command
        cmd = [
    "conda", "run", "-n", "elevate3d", "python", "main_refactor.py",
    f"--obj_name={obj_name}",
    f"--conf_name={conf_name}",
    f"--device_idx={device_idx}",
       ]


        if bake_mesh:
            cmd.append("--bake_mesh")

        # Mode toggles (assumes upstream supports these flags; otherwise ignore)
        if mode == "texture_only":
            cmd.append("--texture_only")
        elif mode == "geometry_only":
            cmd.append("--geometry_only")

        self._run_cmd(cmd, cwd=".")

        # Collect outputs
        out_dir = Path("Outputs/3D") / dataset / obj_name
        if not out_dir.exists():
            # Try alternate default from README (MyData)
            out_dir = Path("Outputs/3D") / "MyData" / obj_name
        if not out_dir.exists():
            raise RuntimeError(f"Expected output dir not found: {out_dir}")

        if return_zip:
            zip_path = Path(f"{obj_name}_results.zip")
            self._zipdir(out_dir, zip_path)
            return CogPath(str(zip_path))

        # If not zipping, try to return a baked mesh if it exists, otherwise the folder
        baked = next(out_dir.glob("**/*.obj"), None) or next(out_dir.glob("**/*.glb"), None)
        if baked:
            return CogPath(str(baked))
        # Last resort: zip folder anyway
        zip_path = Path(f"{obj_name}_results.zip")
        self._zipdir(out_dir, zip_path)
        return CogPath(str(zip_path))

    def _run_cmd(self, cmd, cwd="."):
        print("Running:", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed with code {proc.returncode}:\n{proc.stdout}")
        else:
            lines = proc.stdout.strip().splitlines()
            print("\n".join(lines[-50:]))

    def _zipdir(self, path: Path, zip_path: Path):
        from zipfile import ZipFile, ZIP_DEFLATED
        with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
            for p in path.rglob("*"):
                if p.is_file():
                    zf.write(p, p.relative_to(path))
