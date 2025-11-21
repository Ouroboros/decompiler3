; ===== MLIL Function Dummy_m4000_talk0 @ 0x8B977 =====
; Variables: 10
;
;   var_s0 (slot 0)
;   var_s1 (slot 1)
;   var_s2 (slot 2)
;   var_s3 (slot 3)
;   var_s4 (slot 4)
;   var_s5 (slot 5)
;   var_s6 (slot 6)
;   var_s7 (slot 7)
;   var_s8 (slot 8)
;   var_s9 (slot 9)
;
; Inferred Types:
;   var_s0: int
;   var_s6: float
;   var_s7: float
;   var_s8: float
;   var_s9: int

Dummy_m4000_talk0:
  ; debug.line(10957)
  TALK_BEGIN(0, 4.0)
  goto loc_8B995

loc_8B995:
  ; debug.line(10960)
  menu_create(0, 0, 0, 24)
  goto loc_8B9BF

loc_8B9BF:
  ; debug.line(10963)
  menu_additem(0, "中間地点①", 0)
  goto loc_8B9E3

loc_8B9E3:
  ; debug.line(10965)
  menu_open(0, -1, -1, 1)
  goto loc_8BA0D

loc_8BA0D:
  ; debug.line(10967)
  menu_wait(0)
  goto loc_8BA2B

loc_8BA2B:
  var_s0 = REG[0]
  ; debug.line(10968)
  menu_close(0)
  goto loc_8BA4A

loc_8BA4A:
  ; debug.line(10970)
  if ((var_s0 < 0)) goto loc_8BC2A else loc_8BA5E

loc_8BA5E:
  ; debug.line(10972)
  var_s6 = 0.5
  fade_out(0.5, 0, 1.0, 0)
  goto loc_8BA88

loc_8BA88:
  ; debug.line(10973)
  fade_wait(0)
  goto loc_8BAA0

loc_8BAA0:
  ; debug.line(10975)
  if ((var_s0 != 0)) goto loc_8BB25 else loc_8BAB4

loc_8BAB4:
  ; debug.line(10976)
  chr_set_pos(65000, -259.911, -0.0, 203.559, 172.54)
  goto loc_8BAE4

loc_8BAE4:
  ; debug.line(10977)
  var_s6 = 0.0
  var_s7 = 0.0
  var_s8 = 0.0
  var_s9 = 65000
  camera_rotate_chr(65000, 0.0, 0.0, 0.0, 0, 3, -1)
  goto loc_8BB20

loc_8BB20:
  goto loc_8BC2A

loc_8BB25:
  ; debug.line(10979)
  if ((var_s0 != 1)) goto loc_8BBAA else loc_8BB39

loc_8BB39:
  ; debug.line(10980)
  chr_set_pos(65000, -434.334, 5.358, 42.086, 355.173)
  goto loc_8BB69

loc_8BB69:
  ; debug.line(10981)
  var_s6 = 0.0
  var_s7 = 0.0
  var_s8 = 0.0
  var_s9 = 65000
  camera_rotate_chr(65000, 0.0, 0.0, 0.0, 0, 3, -1)
  goto loc_8BBA5

loc_8BBA5:
  goto loc_8BC2A

loc_8BBAA:
  ; debug.line(10983)
  if ((var_s0 != 2)) goto loc_8BC2A else loc_8BBBE

loc_8BBBE:
  ; debug.line(10984)
  chr_set_pos(65000, -5.976, -0.283, -157.72, 112.963)
  goto loc_8BBEE

loc_8BBEE:
  ; debug.line(10985)
  var_s6 = 0.0
  var_s7 = 0.0
  var_s8 = 0.0
  var_s9 = 65000
  camera_rotate_chr(65000, 0.0, 0.0, 0.0, 0, 3, -1)
  goto loc_8BC2A

loc_8BC2A:
  ; debug.line(10989)
  TALK_END()
  goto loc_8BC3C

loc_8BC3C:
  ; debug.line(10991)
  if ((var_s0 < 0)) goto loc_8BC7A else loc_8BC50

loc_8BC50:
  ; debug.line(10992)
  fade_in(0.5, 0, 0.0, 0)
  goto loc_8BC7A

loc_8BC7A:
  ; debug.line(10995)
  REG[0] = 0
  return


