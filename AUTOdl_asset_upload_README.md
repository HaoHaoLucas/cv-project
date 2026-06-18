# AutoDL asset upload package

Local package:

```text
cv-project-assets-upload.tar
```

Upload it to the AutoDL data disk, for example:

```bash
/root/autodl-tmp/cv-project-assets-upload.tar
```

Then extract it on the server:

```bash
cd /root/autodl-tmp
mkdir -p cv-project-assets
tar xf cv-project-assets-upload.tar -C cv-project-assets
```

After cloning the project:

```bash
cd /root/autodl-tmp
git clone https://github.com/HaoHaoLucas/cv-project.git
cd cv-project
ln -s /root/autodl-tmp/cv-project-assets/data data
ln -s /root/autodl-tmp/cv-project-assets/weights weights
mkdir -p third_party
ln -s /root/autodl-tmp/cv-project-assets/third_party/GroundingDINO third_party/GroundingDINO
```

Smoke test:

```bash
python scripts/setup_env.py
python src/ovd/eval_coco.py --config configs/coco_ovd.yaml --subset 20
python scripts/run_vg_eval.py --dataset refcoco --split validation --max-samples 20 --hf-dir data/refcoco_hf
```

Package contents:

- `data/coco/`
- `data/refcoco_hf/`
- `weights/`
- `third_party/GroundingDINO/`

Excluded from the package:

- raw COCO zip archives
- HuggingFace cache directories
- `third_party/GroundingDINO/.git`
- build/cache artifacts
