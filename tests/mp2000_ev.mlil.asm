; ===== MLIL Function CHR_SETARM @ 0xEF980 =====
; Variables: 8
;
;   arg1
;   arg2
;   arg3
;   var_s3 (slot 3)
;   var_s4 (slot 4)
;   var_s5 (slot 5)
;   var_s6 (slot 6)
;   var_s7 (slot 7)
;
; Inferred Types:
;   arg3: int

CHR_SETARM:
  if ((arg3 != 0)) goto loc_EF9DC else loc_EF991

loc_EF991:
  syscall(1, 47, 0, 2, "AniAttachWeapon", -1, 1.0)
  syscall(1, 47, 0, 2, "AniBtlWait", arg2, arg1)
  goto loc_F0150

loc_EF9DC:
  if ((arg3 != 1)) goto loc_EFA38 else loc_EF9ED

loc_EF9ED:
  syscall(1, 47, 1, 2, "AniAttachWeapon", -1, 1.0)
  syscall(1, 47, 1, 2, "AniBtlWait", arg2, arg1)
  goto loc_F0150

loc_EFA38:
  if ((arg3 != 2)) goto loc_EFA94 else loc_EFA49

loc_EFA49:
  syscall(1, 47, 2, 2, "AniAttachWeapon", -1, 1.0)
  syscall(1, 47, 2, 2, "AniBtlWait", arg2, arg1)
  goto loc_F0150

loc_EFA94:
  if ((arg3 != 3)) goto loc_EFAF0 else loc_EFAA5

loc_EFAA5:
  syscall(1, 47, 3, 2, "AniAttachWeapon", -1, 1.0)
  syscall(1, 47, 3, 2, "AniBtlWait", arg2, arg1)
  goto loc_F0150

loc_EFAF0:
  if ((arg3 != 4)) goto loc_EFB4C else loc_EFB01

loc_EFB01:
  syscall(1, 47, 4, 2, "AniAttachWeapon", -1, 1.0)
  syscall(1, 47, 4, 2, "AniBtlWait", arg2, arg1)
  goto loc_F0150

loc_EFB4C:
  if ((arg3 != 5)) goto loc_EFBA8 else loc_EFB5D

loc_EFB5D:
  syscall(1, 47, 5, 2, "AniAttachWeapon", -1, 1.0)
  syscall(1, 47, 5, 2, "AniBtlWait", arg2, arg1)
  goto loc_F0150

loc_EFBA8:
  if ((arg3 != 6)) goto loc_EFC04 else loc_EFBB9

loc_EFBB9:
  syscall(1, 47, 6, 2, "AniAttachWeapon", -1, 1.0)
  syscall(1, 47, 6, 2, "AniBtlWait", arg2, arg1)
  goto loc_F0150

loc_EFC04:
  if ((arg3 != 7)) goto loc_EFC60 else loc_EFC15

loc_EFC15:
  syscall(1, 47, 7, 2, "AniAttachWeapon", -1, 1.0)
  syscall(1, 47, 7, 2, "AniBtlWait", arg2, arg1)
  goto loc_F0150

loc_EFC60:
  if ((arg3 != 106)) goto loc_EFCBC else loc_EFC71

loc_EFC71:
  syscall(1, 47, 106, 2, "AniAttachWeapon", -1, 1.0)
  syscall(1, 47, 106, 2, "AniBtlWait", arg2, arg1)
  goto loc_F0150

loc_EFCBC:
  if ((arg3 != 105)) goto loc_EFD18 else loc_EFCCD

loc_EFCCD:
  syscall(1, 47, 105, 2, "AniAttachWeapon", -1, 1.0)
  syscall(1, 47, 105, 2, "AniBtlWait", arg2, arg1)
  goto loc_F0150

loc_EFD18:
  if ((arg3 != 104)) goto loc_EFD74 else loc_EFD29

loc_EFD29:
  syscall(1, 47, 104, 2, "AniAttachWeapon", -1, 1.0)
  syscall(1, 47, 104, 2, "AniBtlWait", arg2, arg1)
  goto loc_F0150

loc_EFD74:
  if ((arg3 != 118)) goto loc_EFDD0 else loc_EFD85

loc_EFD85:
  syscall(1, 47, 118, 2, "AniAttachWeapon", -1, 1.0)
  syscall(1, 47, 118, 2, "AniBtlWait", arg2, arg1)
  goto loc_F0150

loc_EFDD0:
  if ((arg3 != 114)) goto loc_EFE2C else loc_EFDE1

loc_EFDE1:
  syscall(1, 47, 114, 2, "AniAttachWeapon", -1, 1.0)
  syscall(1, 47, 114, 2, "AniBtlWait", arg2, arg1)
  goto loc_F0150

loc_EFE2C:
  if ((arg3 != 108)) goto loc_EFE88 else loc_EFE3D

loc_EFE3D:
  syscall(1, 47, 108, 2, "AniAttachWeapon", -1, 1.0)
  syscall(1, 47, 108, 2, "AniBtlWait", arg2, arg1)
  goto loc_F0150

loc_EFE88:
  if ((arg3 != 101)) goto loc_EFEE4 else loc_EFE99

loc_EFE99:
  syscall(1, 47, 101, 2, "AniAttachWeapon", -1, 1.0)
  syscall(1, 47, 101, 2, "AniBtlWait", arg2, arg1)
  goto loc_F0150

loc_EFEE4:
  if ((arg3 != 102)) goto loc_EFF40 else loc_EFEF5

loc_EFEF5:
  syscall(1, 47, 102, 2, "AniAttachWeapon", -1, 1.0)
  syscall(1, 47, 102, 2, "AniBtlWait", arg2, arg1)
  goto loc_F0150

loc_EFF40:
  if ((arg3 != 103)) goto loc_EFF9C else loc_EFF51

loc_EFF51:
  syscall(1, 47, 103, 2, "AniAttachWeapon", -1, 1.0)
  syscall(1, 47, 103, 2, "AniBtlWait", arg2, arg1)
  goto loc_F0150

loc_EFF9C:
  if ((arg3 != 100)) goto loc_EFFF8 else loc_EFFAD

loc_EFFAD:
  syscall(1, 47, 100, 2, "AniAttachWeapon", -1, 1.0)
  syscall(1, 47, 100, 2, "AniBtlWait", arg2, arg1)
  goto loc_F0150

loc_EFFF8:
  if ((arg3 != 137)) goto loc_F0054 else loc_F0009

loc_F0009:
  syscall(1, 47, 137, 2, "AniAttachWeapon", -1, 1.0)
  syscall(1, 47, 137, 2, "AniBtlWait", arg2, arg1)
  goto loc_F0150

loc_F0054:
  if ((arg3 != 138)) goto loc_F00B0 else loc_F0065

loc_F0065:
  syscall(1, 47, 138, 2, "AniAttachWeapon", -1, 1.0)
  syscall(1, 47, 138, 2, "AniBtlWait", arg2, arg1)
  goto loc_F0150

loc_F00B0:
  if ((arg3 != 139)) goto loc_F010C else loc_F00C1

loc_F00C1:
  syscall(1, 47, 139, 2, "AniAttachWeapon", -1, 1.0)
  syscall(1, 47, 139, 2, "AniBtlWait", arg2, arg1)
  goto loc_F0150

loc_F010C:
  syscall(1, 47, arg3, 2, "AniAttachWeapon", -1, 1.0)
  syscall(1, 47, arg3, 2, "AniEvBtlWait", arg2, arg1)
  goto loc_F0150

loc_F0150:
  REG[0] = 0
  return


; ===== MLIL Function EV_02_63_02_SELECT_00 @ 0x15FBC7 =====
; Variables: 7
;
;   var_s0 (slot 0)
;   var_s1 (slot 1)
;   var_s2 (slot 2)
;   var_s3 (slot 3)
;   var_s4 (slot 4)
;   var_s5 (slot 5)
;   var_s6 (slot 6)
;
; Inferred Types:
;   var_s0: int
;   var_s3: int
;   var_s4: bool
;   var_s5: int
;   var_s6: bool

EV_02_63_02_SELECT_00:
  ; debug.line(26456)
  global_work(0)
  goto loc_15FBDF

loc_15FBDF:
  if ((REG[0] != -1)) goto loc_15FD2D else loc_15FBED

loc_15FBED:
  ; debug.line(26457)
  menu_create(0, 2, 0, 28)
  goto loc_15FC17

loc_15FC17:
  ; debug.line(26458)
  menu_additem(0, "那好，我也去！", 0)
  goto loc_15FC3B

loc_15FC3B:
  ; debug.line(26459)
  menu_additem(0, "这、这家伙也太乱来了……", 1)
  goto loc_15FC5F

loc_15FC5F:
  ; debug.line(26461)
  menu_open(0, -1, -1, 0, "")
  goto loc_15FC8F

loc_15FC8F:
  ; debug.line(26464)
  menu_wait(0)
  goto loc_15FCAD

loc_15FCAD:
  var_s0 = REG[0]
  ; debug.line(26465)
  menu_close(0)
  goto loc_15FCCC

loc_15FCCC:
  ; debug.line(26467)
  menu_is_canceled(0)
  goto loc_15FCE4

loc_15FCE4:
  if (!REG[0]) goto loc_15FD0E else loc_15FCEB

loc_15FCEB:
  ; debug.line(26468)
  set_global_work(0, 1)
  goto loc_15FD09

loc_15FD09:
  goto loc_15FD2B

loc_15FD0E:
  ; debug.line(26470)
  set_global_work(0, var_s0)
  goto loc_15FD2B

loc_15FD2B:
  goto loc_15FD2D

loc_15FD2D:
  ; debug.line(26473)
  REG[0] = 0
  return


