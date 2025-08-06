import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from cog import BasePredictor, Input, Path as CogPath

class Predictor(BasePredictor):
    def setup(self):
        # Validate that required files are present
        required = ["main_refactor.py", "requirements.txt"]
        for f in required:
            if not Path(f).exists():
                raise RuntimeError(f"Required file missing: {f}. Make sure this Predictor lives at the root of Elevate3D.")

        # Ensure PoissonRecon binary exists. If missing, copy it from PATH (built during container build).
        pr_bin = Path("PoissonRecon/Bin/Linux/PoissonRecon")
        if not pr_bin.exists():
            alt = shutil.which("PoissonRecon")
            if alt:
                dest_dir = pr_bin.parent
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(alt, pr_bin)
                os.chmod(pr_bin, 0o755)
                print(f"Copied PoissonRecon binary from {alt} to {pr_bin}.")
            else:
                print(
                    "Warning: PoissonRecon executable not found in repository or on PATH. "
                    "Geometry refinement might fail unless built in the container build step."
                )

        # Download SAM checkpoint if missing
        sam_dir = Path("Checkpoints/sam")
        sam_checkpoint = sam_dir / "sam_vit_h_4b8939.pth"
        if not sam_checkpoint.exists():
            sam_dir.mkdir(parents=True, exist_ok=True)
            try:
                from urllib.request import urlretrieve
                print("Downloading SAM checkpoint...")
                urlretrieve(
                    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                    str(sam_checkpoint),
                )
                print("Downloaded SAM checkpoint.")
            except Exception as e:
                print(f"Warning: failed to download SAM checkpoint: {e}")

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
        # Validate input model format
        model_path = Path(model_file)
        model_ext = model_path.suffix.lower()
        if model_ext not in [".obj", ".glb", ".ply"]:
            raise ValueError("Unsupported model format. Please upload .obj, .glb, or .ply")

        # Prepare inputs directory structure
        in_root = Path("Inputs/3D") / dataset / obj_name
        in_root.mkdir(parents=True, exist_ok=True)
        target_model = in_root / f"{obj_name}{model_ext}"
        shutil.copyfile(model_path, target_model)

        # Save prompt text if provided
        if prompt_text:
            (in_root / "prompt.txt").write_text(prompt_text)

        # Run preprocessing (if available)
        preprocess_script = Path("run_preprocess_script.sh")
        if preprocess_script.exists():
            self._run_cmd(["bash", str(preprocess_script)], cwd=".")
        else:
            if Path("preprocess.py").exists():
                self._run_cmd(["python", "preprocess.py", "--obj_name", obj_name, "--dataset", dataset], cwd=".")
            else:
                print("Preprocess helpers not found. Proceeding directly to refinement...")

        # Build command for refinement step
        cmd = [
            "python", "main_refactor.py",
            f"--obj_name={obj_name}",
            f"--conf_name={conf_name}",
            f"--device_idx={device_idx}",
        ]
        if bake_mesh:
            cmd.append("--bake_mesh")
        if mode == "texture_only":
            cmd.append("--texture_only")
        elif mode == "geometry_only":
            cmd.append("--geometry_only")

        self._run_cmd(cmd, cwd=".")

        # Collect outputs
        out_dir = Path("Outputs/3D") / dataset / obj_name
        if not out_dir.exists():
            out_dir = Path("Outputs/3D") / "MyData" / obj_name
        if not out_dir.exists():
            raise RuntimeError(f"Expected output dir not found: {out_dir}")

        if return_zip:
            zip_path = Path(f"{obj_name}_results.zip")
            self._zipdir(out_dir, zip_path)
            return CogPath(str(zip_path))

        baked = next(out_dir.glob("**/*.obj"), None) or next(out_dir.glob("**/*.glb"), None)
        if baked:
            return CogPath(str(baked))
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
