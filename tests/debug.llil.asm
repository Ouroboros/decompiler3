; ---------- Dummy_m4000_talk0 ----------
block_0(0x8B977), Dummy_m4000_talk0, [sp = 0, fp = 0]
  dbg_line 10957
  STACK[sp] = <func_id> ; [0]
  sp++
  STACK[sp] = <&loc_8B995> ; [1]
  sp++
  STACK[sp] = 4 ; [2]
  sp++
  STACK[sp] = 0 ; [3]
  sp++
  call TALK_BEGIN(0, 4)

block_1(0x8B995), loc_8B995, [sp = 0]
  dbg_line 10960
  STACK[sp] = <func_id> ; [0]
  sp++
  STACK[sp] = <&loc_8B9BF> ; [1]
  sp++
  STACK[sp] = 24 ; [2]
  sp++
  STACK[sp] = 0 ; [3]
  sp++
  STACK[sp] = 0 ; [4]
  sp++
  STACK[sp] = 0 ; [5]
  sp++
  call menu_create(0, 0, 0, 24)

block_2(0x8B9BF), loc_8B9BF, [sp = 0]
  dbg_line 10963
  STACK[sp] = <func_id> ; [0]
  sp++
  STACK[sp] = <&loc_8B9E3> ; [1]
  sp++
  STACK[sp] = 0 ; [2]
  sp++
  STACK[sp] = '中間地点①' ; [3]
  sp++
  STACK[sp] = 0 ; [4]
  sp++
  call menu_additem(0, '中間地点①', 0)

block_3(0x8B9E3), loc_8B9E3, [sp = 0]
  dbg_line 10965
  STACK[sp] = <func_id> ; [0]
  sp++
  STACK[sp] = <&loc_8BA0D> ; [1]
  sp++
  STACK[sp] = 1 ; [2]
  sp++
  STACK[sp] = -1 ; [3]
  sp++
  STACK[sp] = -1 ; [4]
  sp++
  STACK[sp] = 0 ; [5]
  sp++
  call menu_open(0, -1, -1, 1)

block_4(0x8BA0D), loc_8BA0D, [sp = 0]
  dbg_line 10967
  STACK[sp] = 0x0 ; [0]
  sp++
  STACK[sp] = <func_id> ; [1]
  sp++
  STACK[sp] = <&loc_8BA2B> ; [2]
  sp++
  STACK[sp] = 0 ; [3]
  sp++
  call menu_wait(0)

block_5(0x8BA2B), loc_8BA2B, [sp = 1]
  STACK[sp] = REG[0] ; [1]
  sp++
  STACK[0] = STACK[--sp] ; REG[0]

  dbg_line 10968
  STACK[sp] = <func_id> ; [1]
  sp++
  STACK[sp] = <&loc_8BA4A> ; [2]
  sp++
  STACK[sp] = 0 ; [3]
  sp++
  call menu_close(0)

block_6(0x8BA4A), loc_8BA4A, [sp = 1]
  dbg_line 10970
  STACK[sp] = STACK[sp - 1<0>] ; [1]
  sp++
  STACK[sp] = 0 ; [2]
  sp++
  rhs = STACK[--sp]  ; 0
  lhs = STACK[--sp]  ; STACK[sp - 1<0>]
  STACK[sp++] = (lhs >= rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_20 else block_7

block_7(0x8BA5E), loc_8BA5E, [sp = 1]
  dbg_line 10972
  STACK[sp] = <func_id> ; [1]
  sp++
  STACK[sp] = <&loc_8BA88> ; [2]
  sp++
  STACK[sp] = 0 ; [3]
  sp++
  STACK[sp] = 1 ; [4]
  sp++
  STACK[sp] = 0 ; [5]
  sp++
  STACK[sp] = 0.5 ; [6]
  sp++
  call fade_out(0.5, 0, 1, 0)

block_8(0x8BA88), loc_8BA88, [sp = 1]
  dbg_line 10973
  STACK[sp] = <func_id> ; [1]
  sp++
  STACK[sp] = <&loc_8BAA0> ; [2]
  sp++
  STACK[sp] = 0 ; [3]
  sp++
  call fade_wait(0)

block_9(0x8BAA0), loc_8BAA0, [sp = 1]
  dbg_line 10975
  STACK[sp] = STACK[sp - 1<0>] ; [1]
  sp++
  STACK[sp] = 0 ; [2]
  sp++
  rhs = STACK[--sp]  ; 0
  lhs = STACK[--sp]  ; STACK[sp - 1<0>]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_13 else block_10

block_10(0x8BAB4), loc_8BAB4, [sp = 1]
  dbg_line 10976
  STACK[sp] = <func_id> ; [1]
  sp++
  STACK[sp] = <&loc_8BAE4> ; [2]
  sp++
  STACK[sp] = 172.539978 ; [3]
  sp++
  STACK[sp] = 203.55896 ; [4]
  sp++
  STACK[sp] = -0 ; [5]
  sp++
  STACK[sp] = -259.911011 ; [6]
  sp++
  STACK[sp] = 65000 ; [7]
  sp++
  call chr_set_pos(65000, -259.911011, -0, 203.55896, 172.539978)

block_11(0x8BAE4), loc_8BAE4, [sp = 1]
  dbg_line 10977
  STACK[sp] = <func_id> ; [1]
  sp++
  STACK[sp] = <&loc_8BB20> ; [2]
  sp++
  STACK[sp] = -1 ; [3]
  sp++
  STACK[sp] = 3 ; [4]
  sp++
  STACK[sp] = 0 ; [5]
  sp++
  STACK[sp] = 0 ; [6]
  sp++
  STACK[sp] = 0 ; [7]
  sp++
  STACK[sp] = 0 ; [8]
  sp++
  STACK[sp] = 65000 ; [9]
  sp++
  call camera_rotate_chr(65000, 0, 0, 0, 0, 3, -1)

block_12(0x8BB20), loc_8BB20, [sp = 1]
  goto block_20

block_13(0x8BB25), loc_8BB25, [sp = 1]
  dbg_line 10979
  STACK[sp] = STACK[sp - 1<0>] ; [1]
  sp++
  STACK[sp] = 1 ; [2]
  sp++
  rhs = STACK[--sp]  ; 1
  lhs = STACK[--sp]  ; STACK[sp - 1<0>]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_17 else block_14

block_14(0x8BB39), loc_8BB39, [sp = 1]
  dbg_line 10980
  STACK[sp] = <func_id> ; [1]
  sp++
  STACK[sp] = <&loc_8BB69> ; [2]
  sp++
  STACK[sp] = 355.172974 ; [3]
  sp++
  STACK[sp] = 42.085999 ; [4]
  sp++
  STACK[sp] = 5.358 ; [5]
  sp++
  STACK[sp] = -434.333984 ; [6]
  sp++
  STACK[sp] = 65000 ; [7]
  sp++
  call chr_set_pos(65000, -434.333984, 5.358, 42.085999, 355.172974)

block_15(0x8BB69), loc_8BB69, [sp = 1]
  dbg_line 10981
  STACK[sp] = <func_id> ; [1]
  sp++
  STACK[sp] = <&loc_8BBA5> ; [2]
  sp++
  STACK[sp] = -1 ; [3]
  sp++
  STACK[sp] = 3 ; [4]
  sp++
  STACK[sp] = 0 ; [5]
  sp++
  STACK[sp] = 0 ; [6]
  sp++
  STACK[sp] = 0 ; [7]
  sp++
  STACK[sp] = 0 ; [8]
  sp++
  STACK[sp] = 65000 ; [9]
  sp++
  call camera_rotate_chr(65000, 0, 0, 0, 0, 3, -1)

block_16(0x8BBA5), loc_8BBA5, [sp = 1]
  goto block_20

block_17(0x8BBAA), loc_8BBAA, [sp = 1]
  dbg_line 10983
  STACK[sp] = STACK[sp - 1<0>] ; [1]
  sp++
  STACK[sp] = 2 ; [2]
  sp++
  rhs = STACK[--sp]  ; 2
  lhs = STACK[--sp]  ; STACK[sp - 1<0>]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_20 else block_18

block_18(0x8BBBE), loc_8BBBE, [sp = 1]
  dbg_line 10984
  STACK[sp] = <func_id> ; [1]
  sp++
  STACK[sp] = <&loc_8BBEE> ; [2]
  sp++
  STACK[sp] = 112.962982 ; [3]
  sp++
  STACK[sp] = -157.719971 ; [4]
  sp++
  STACK[sp] = -0.283 ; [5]
  sp++
  STACK[sp] = -5.976 ; [6]
  sp++
  STACK[sp] = 65000 ; [7]
  sp++
  call chr_set_pos(65000, -5.976, -0.283, -157.719971, 112.962982)

block_19(0x8BBEE), loc_8BBEE, [sp = 1]
  dbg_line 10985
  STACK[sp] = <func_id> ; [1]
  sp++
  STACK[sp] = <&loc_8BC2A> ; [2]
  sp++
  STACK[sp] = -1 ; [3]
  sp++
  STACK[sp] = 3 ; [4]
  sp++
  STACK[sp] = 0 ; [5]
  sp++
  STACK[sp] = 0 ; [6]
  sp++
  STACK[sp] = 0 ; [7]
  sp++
  STACK[sp] = 0 ; [8]
  sp++
  STACK[sp] = 65000 ; [9]
  sp++
  call camera_rotate_chr(65000, 0, 0, 0, 0, 3, -1)

block_20(0x8BC2A), loc_8BC2A, [sp = 1]
  dbg_line 10989
  STACK[sp] = <func_id> ; [1]
  sp++
  STACK[sp] = <&loc_8BC3C> ; [2]
  sp++
  call TALK_END

block_21(0x8BC3C), loc_8BC3C, [sp = 1]
  dbg_line 10991
  STACK[sp] = STACK[sp - 1<0>] ; [1]
  sp++
  STACK[sp] = 0 ; [2]
  sp++
  rhs = STACK[--sp]  ; 0
  lhs = STACK[--sp]  ; STACK[sp - 1<0>]
  STACK[sp++] = (lhs >= rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_23 else block_22

block_22(0x8BC50), loc_8BC50, [sp = 1]
  dbg_line 10992
  STACK[sp] = <func_id> ; [1]
  sp++
  STACK[sp] = <&loc_8BC7A> ; [2]
  sp++
  STACK[sp] = 0 ; [3]
  sp++
  STACK[sp] = 0 ; [4]
  sp++
  STACK[sp] = 0 ; [5]
  sp++
  STACK[sp] = 0.5 ; [6]
  sp++
  call fade_in(0.5, 0, 0, 0)

block_23(0x8BC7A), loc_8BC7A, [sp = 1]
  dbg_line 10995
  STACK[sp] = 0x0 ; [1]
  sp++
  REG[0] = STACK[--sp]  ; 0x0
  sp--
  return


