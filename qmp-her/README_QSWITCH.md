# Q-Switch Primitive Training

Kode di folder ini mengadaptasi ide QMP-HER paper untuk repo ini tanpa mengubah
`source/envs` atau `script/train_sac.py`.

## Komponen

- `qmpher/q_switch_sac.py`: subclass `SAC` yang memilih aksi dengan target critic.
- `qmpher/primitives.py`: adapter kandidat aksi dari model `GraspingEnvIK` dan `InsertTargetEnvIK`.
- `qmpher/sync.py`: sinkronisasi state target env ke hidden `GraspingEnvIK` agar observation primitive grasp tetap valid.
- `qmpher/envs.py`: reset end-to-end lokal dan wrapper gripper manual/reward release.
- `train_qswitch.py`: entrypoint training.

## Alur Training

Pada setiap step rollout:

1. primitive grasp memberi kandidat aksi 6-DoF;
2. primitive insert memberi kandidat aksi 6-DoF;
3. target SAC policy memberi kandidat aksi 6-DoF setelah warmup;
4. critic target policy menghitung `Q(s, a)` untuk semua kandidat;
5. aksi dengan Q tertinggi dieksekusi.

Gripper tidak menjadi bagian action policy. Wrapper manual:

- membuka gripper saat reset;
- menutup gripper saat EE dekat objek;
- membuka gripper saat objek sudah align dengan target insert;
- setelah release, reward posisi/orientasi terhadap target 5 cm di atas place bisa dinonaktifkan dan diganti bonus release.

## Contoh Run

```bash
python qmp-her/train_qswitch.py \
  --grasp-model melogs/ik_models/grasp-ik-model.zip \
  --insert-model melogs/ik_models/insert-ik-model.zip \
  --total-timesteps 1000000
```

Untuk memakai reset asli `InsertTargetEnvIK` yang mulai dari snapshot grasp:

```bash
python qmp-her/train_qswitch.py --regular-insert-reset
```

Default-nya memakai `QMPInsertEndToEndEnv`, yaitu reset objek di meja supaya
primitive grasp benar-benar ikut berperan.
