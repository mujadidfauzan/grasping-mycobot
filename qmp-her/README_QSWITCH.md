# Q-Switch Primitive Training

Kode di folder ini mengadaptasi ide QMP-HER paper untuk repo ini tanpa mengubah
`source/envs` atau `script/train_sac.py`.

## Komponen

- `qmpher/q_switch_sac.py`: subclass `SAC` yang memilih aksi dengan target critic.
- `qmpher/primitives.py`: adapter kandidat aksi dari model `GraspingEnvIK` dan `InsertTargetEnvIK`.
- `qmpher/sync.py`: sinkronisasi state target env ke hidden `GraspingEnvIK` agar observation primitive grasp tetap valid.
- `qmpher/envs.py`: env mandiri `QMPGraspInsertEnv` berbasis `MujocoEnv`, goal-conditioned untuk HER.
- `train_qswitch.py`: entrypoint training.

## Alur Training

Pada setiap step rollout:

1. primitive grasp memberi kandidat aksi 6-DoF;
2. primitive insert memberi kandidat aksi 6-DoF;
3. target SAC policy memberi kandidat aksi 6-DoF setelah warmup;
4. critic target policy menghitung `Q(s, a)` untuk semua kandidat;
5. aksi dengan Q tertinggi dieksekusi;
6. transisi disimpan ke `HerReplayBuffer` dan goal direlabel dengan strategi HER.

Training memakai `MultiInputPolicy` dan observation dict:

- `observation`: state flat yang juga dipakai adapter primitive insert;
- `achieved_goal`: pose objek `[x, y, z, qw, qx, qy, qz]`;
- `desired_goal`: pose target place aktual `[x, y, z, qw, qx, qy, qz]`.

Gripper tidak menjadi bagian action policy. Env mandiri:

- membuka gripper saat reset;
- menutup gripper saat EE dekat objek;
- membuka gripper saat objek sudah align dengan target insert di atas place;
- reward `0` hanya ketika pose objek sudah masuk ke target place
  (`cube_place_site`), dengan radial dan tinggi default `8 mm`; pose di atas place
  hanya menjadi trigger release.

## Contoh Run

```bash
python qmp-her/train_qswitch.py \
  --grasp-model melogs/ik_models/grasp-ik-model.zip \
  --insert-model melogs/ik_models/insert-ik-model.zip \
  --total-timesteps 1000000
```

Training env selalu reset objek di meja supaya primitive grasp benar-benar ikut
berperan. Primitive insert tetap boleh berasal dari model `InsertTargetEnvIK`,
tetapi env target Q-switch tidak mewarisi class `InsertTargetEnvIK`.

Secara default, setelah 1.000.000 timestep rollout berpindah penuh ke target
policy (`--target-policy-only-starts 1000000`), sehingga primitive dan scripted
lift tidak lagi dipakai untuk memilih aksi.

Video rollout otomatis direkam setiap 50.000 step ke `runs/<run-name>/videos`.
Gunakan `--video-freq 0` untuk mematikan recording, atau ubah interval dengan
`--video-freq`.
