; ---------- CHR_SETARM ----------
block_0(0xEF980), CHR_SETARM, [sp = 3, fp = 0]
  STACK[sp] = STACK[fp + 2] ; [3]
  sp++
  STACK[sp] = 0 ; [4]
  sp++
  rhs = STACK[--sp]  ; 0
  lhs = STACK[--sp]  ; STACK[fp + 2]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_2 else block_1

block_1(0xEF991), loc_EF991, [sp = 3]
  STACK[sp] = 1 ; [3]
  sp++
  STACK[sp] = -1 ; [4]
  sp++
  STACK[sp] = 'AniAttachWeapon' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 0 ; [7]
  sp++
  SYSCALL(1, 0x2f, 0, 2, 'AniAttachWeapon', -1, 1)
  sp -= 5
  STACK[sp] = STACK[fp + 0] ; [3]
  sp++
  STACK[sp] = STACK[fp + 1] ; [4]
  sp++
  STACK[sp] = 'AniBtlWait' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 0 ; [7]
  sp++
  SYSCALL(1, 0x2f, 0, 2, 'AniBtlWait', STACK[fp + 1], STACK[fp + 0])
  sp -= 5
  goto block_43

block_2(0xEF9DC), loc_EF9DC, [sp = 3]
  STACK[sp] = STACK[fp + 2] ; [3]
  sp++
  STACK[sp] = 1 ; [4]
  sp++
  rhs = STACK[--sp]  ; 1
  lhs = STACK[--sp]  ; STACK[fp + 2]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_4 else block_3

block_3(0xEF9ED), loc_EF9ED, [sp = 3]
  STACK[sp] = 1 ; [3]
  sp++
  STACK[sp] = -1 ; [4]
  sp++
  STACK[sp] = 'AniAttachWeapon' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 1 ; [7]
  sp++
  SYSCALL(1, 0x2f, 1, 2, 'AniAttachWeapon', -1, 1)
  sp -= 5
  STACK[sp] = STACK[fp + 0] ; [3]
  sp++
  STACK[sp] = STACK[fp + 1] ; [4]
  sp++
  STACK[sp] = 'AniBtlWait' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 1 ; [7]
  sp++
  SYSCALL(1, 0x2f, 1, 2, 'AniBtlWait', STACK[fp + 1], STACK[fp + 0])
  sp -= 5
  goto block_43

block_4(0xEFA38), loc_EFA38, [sp = 3]
  STACK[sp] = STACK[fp + 2] ; [3]
  sp++
  STACK[sp] = 2 ; [4]
  sp++
  rhs = STACK[--sp]  ; 2
  lhs = STACK[--sp]  ; STACK[fp + 2]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_6 else block_5

block_5(0xEFA49), loc_EFA49, [sp = 3]
  STACK[sp] = 1 ; [3]
  sp++
  STACK[sp] = -1 ; [4]
  sp++
  STACK[sp] = 'AniAttachWeapon' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 2 ; [7]
  sp++
  SYSCALL(1, 0x2f, 2, 2, 'AniAttachWeapon', -1, 1)
  sp -= 5
  STACK[sp] = STACK[fp + 0] ; [3]
  sp++
  STACK[sp] = STACK[fp + 1] ; [4]
  sp++
  STACK[sp] = 'AniBtlWait' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 2 ; [7]
  sp++
  SYSCALL(1, 0x2f, 2, 2, 'AniBtlWait', STACK[fp + 1], STACK[fp + 0])
  sp -= 5
  goto block_43

block_6(0xEFA94), loc_EFA94, [sp = 3]
  STACK[sp] = STACK[fp + 2] ; [3]
  sp++
  STACK[sp] = 3 ; [4]
  sp++
  rhs = STACK[--sp]  ; 3
  lhs = STACK[--sp]  ; STACK[fp + 2]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_8 else block_7

block_7(0xEFAA5), loc_EFAA5, [sp = 3]
  STACK[sp] = 1 ; [3]
  sp++
  STACK[sp] = -1 ; [4]
  sp++
  STACK[sp] = 'AniAttachWeapon' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 3 ; [7]
  sp++
  SYSCALL(1, 0x2f, 3, 2, 'AniAttachWeapon', -1, 1)
  sp -= 5
  STACK[sp] = STACK[fp + 0] ; [3]
  sp++
  STACK[sp] = STACK[fp + 1] ; [4]
  sp++
  STACK[sp] = 'AniBtlWait' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 3 ; [7]
  sp++
  SYSCALL(1, 0x2f, 3, 2, 'AniBtlWait', STACK[fp + 1], STACK[fp + 0])
  sp -= 5
  goto block_43

block_8(0xEFAF0), loc_EFAF0, [sp = 3]
  STACK[sp] = STACK[fp + 2] ; [3]
  sp++
  STACK[sp] = 4 ; [4]
  sp++
  rhs = STACK[--sp]  ; 4
  lhs = STACK[--sp]  ; STACK[fp + 2]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_10 else block_9

block_9(0xEFB01), loc_EFB01, [sp = 3]
  STACK[sp] = 1 ; [3]
  sp++
  STACK[sp] = -1 ; [4]
  sp++
  STACK[sp] = 'AniAttachWeapon' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 4 ; [7]
  sp++
  SYSCALL(1, 0x2f, 4, 2, 'AniAttachWeapon', -1, 1)
  sp -= 5
  STACK[sp] = STACK[fp + 0] ; [3]
  sp++
  STACK[sp] = STACK[fp + 1] ; [4]
  sp++
  STACK[sp] = 'AniBtlWait' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 4 ; [7]
  sp++
  SYSCALL(1, 0x2f, 4, 2, 'AniBtlWait', STACK[fp + 1], STACK[fp + 0])
  sp -= 5
  goto block_43

block_10(0xEFB4C), loc_EFB4C, [sp = 3]
  STACK[sp] = STACK[fp + 2] ; [3]
  sp++
  STACK[sp] = 5 ; [4]
  sp++
  rhs = STACK[--sp]  ; 5
  lhs = STACK[--sp]  ; STACK[fp + 2]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_12 else block_11

block_11(0xEFB5D), loc_EFB5D, [sp = 3]
  STACK[sp] = 1 ; [3]
  sp++
  STACK[sp] = -1 ; [4]
  sp++
  STACK[sp] = 'AniAttachWeapon' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 5 ; [7]
  sp++
  SYSCALL(1, 0x2f, 5, 2, 'AniAttachWeapon', -1, 1)
  sp -= 5
  STACK[sp] = STACK[fp + 0] ; [3]
  sp++
  STACK[sp] = STACK[fp + 1] ; [4]
  sp++
  STACK[sp] = 'AniBtlWait' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 5 ; [7]
  sp++
  SYSCALL(1, 0x2f, 5, 2, 'AniBtlWait', STACK[fp + 1], STACK[fp + 0])
  sp -= 5
  goto block_43

block_12(0xEFBA8), loc_EFBA8, [sp = 3]
  STACK[sp] = STACK[fp + 2] ; [3]
  sp++
  STACK[sp] = 6 ; [4]
  sp++
  rhs = STACK[--sp]  ; 6
  lhs = STACK[--sp]  ; STACK[fp + 2]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_14 else block_13

block_13(0xEFBB9), loc_EFBB9, [sp = 3]
  STACK[sp] = 1 ; [3]
  sp++
  STACK[sp] = -1 ; [4]
  sp++
  STACK[sp] = 'AniAttachWeapon' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 6 ; [7]
  sp++
  SYSCALL(1, 0x2f, 6, 2, 'AniAttachWeapon', -1, 1)
  sp -= 5
  STACK[sp] = STACK[fp + 0] ; [3]
  sp++
  STACK[sp] = STACK[fp + 1] ; [4]
  sp++
  STACK[sp] = 'AniBtlWait' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 6 ; [7]
  sp++
  SYSCALL(1, 0x2f, 6, 2, 'AniBtlWait', STACK[fp + 1], STACK[fp + 0])
  sp -= 5
  goto block_43

block_14(0xEFC04), loc_EFC04, [sp = 3]
  STACK[sp] = STACK[fp + 2] ; [3]
  sp++
  STACK[sp] = 7 ; [4]
  sp++
  rhs = STACK[--sp]  ; 7
  lhs = STACK[--sp]  ; STACK[fp + 2]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_16 else block_15

block_15(0xEFC15), loc_EFC15, [sp = 3]
  STACK[sp] = 1 ; [3]
  sp++
  STACK[sp] = -1 ; [4]
  sp++
  STACK[sp] = 'AniAttachWeapon' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 7 ; [7]
  sp++
  SYSCALL(1, 0x2f, 7, 2, 'AniAttachWeapon', -1, 1)
  sp -= 5
  STACK[sp] = STACK[fp + 0] ; [3]
  sp++
  STACK[sp] = STACK[fp + 1] ; [4]
  sp++
  STACK[sp] = 'AniBtlWait' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 7 ; [7]
  sp++
  SYSCALL(1, 0x2f, 7, 2, 'AniBtlWait', STACK[fp + 1], STACK[fp + 0])
  sp -= 5
  goto block_43

block_16(0xEFC60), loc_EFC60, [sp = 3]
  STACK[sp] = STACK[fp + 2] ; [3]
  sp++
  STACK[sp] = 106 ; [4]
  sp++
  rhs = STACK[--sp]  ; 106
  lhs = STACK[--sp]  ; STACK[fp + 2]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_18 else block_17

block_17(0xEFC71), loc_EFC71, [sp = 3]
  STACK[sp] = 1 ; [3]
  sp++
  STACK[sp] = -1 ; [4]
  sp++
  STACK[sp] = 'AniAttachWeapon' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 106 ; [7]
  sp++
  SYSCALL(1, 0x2f, 106, 2, 'AniAttachWeapon', -1, 1)
  sp -= 5
  STACK[sp] = STACK[fp + 0] ; [3]
  sp++
  STACK[sp] = STACK[fp + 1] ; [4]
  sp++
  STACK[sp] = 'AniBtlWait' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 106 ; [7]
  sp++
  SYSCALL(1, 0x2f, 106, 2, 'AniBtlWait', STACK[fp + 1], STACK[fp + 0])
  sp -= 5
  goto block_43

block_18(0xEFCBC), loc_EFCBC, [sp = 3]
  STACK[sp] = STACK[fp + 2] ; [3]
  sp++
  STACK[sp] = 105 ; [4]
  sp++
  rhs = STACK[--sp]  ; 105
  lhs = STACK[--sp]  ; STACK[fp + 2]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_20 else block_19

block_19(0xEFCCD), loc_EFCCD, [sp = 3]
  STACK[sp] = 1 ; [3]
  sp++
  STACK[sp] = -1 ; [4]
  sp++
  STACK[sp] = 'AniAttachWeapon' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 105 ; [7]
  sp++
  SYSCALL(1, 0x2f, 105, 2, 'AniAttachWeapon', -1, 1)
  sp -= 5
  STACK[sp] = STACK[fp + 0] ; [3]
  sp++
  STACK[sp] = STACK[fp + 1] ; [4]
  sp++
  STACK[sp] = 'AniBtlWait' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 105 ; [7]
  sp++
  SYSCALL(1, 0x2f, 105, 2, 'AniBtlWait', STACK[fp + 1], STACK[fp + 0])
  sp -= 5
  goto block_43

block_20(0xEFD18), loc_EFD18, [sp = 3]
  STACK[sp] = STACK[fp + 2] ; [3]
  sp++
  STACK[sp] = 104 ; [4]
  sp++
  rhs = STACK[--sp]  ; 104
  lhs = STACK[--sp]  ; STACK[fp + 2]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_22 else block_21

block_21(0xEFD29), loc_EFD29, [sp = 3]
  STACK[sp] = 1 ; [3]
  sp++
  STACK[sp] = -1 ; [4]
  sp++
  STACK[sp] = 'AniAttachWeapon' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 104 ; [7]
  sp++
  SYSCALL(1, 0x2f, 104, 2, 'AniAttachWeapon', -1, 1)
  sp -= 5
  STACK[sp] = STACK[fp + 0] ; [3]
  sp++
  STACK[sp] = STACK[fp + 1] ; [4]
  sp++
  STACK[sp] = 'AniBtlWait' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 104 ; [7]
  sp++
  SYSCALL(1, 0x2f, 104, 2, 'AniBtlWait', STACK[fp + 1], STACK[fp + 0])
  sp -= 5
  goto block_43

block_22(0xEFD74), loc_EFD74, [sp = 3]
  STACK[sp] = STACK[fp + 2] ; [3]
  sp++
  STACK[sp] = 118 ; [4]
  sp++
  rhs = STACK[--sp]  ; 118
  lhs = STACK[--sp]  ; STACK[fp + 2]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_24 else block_23

block_23(0xEFD85), loc_EFD85, [sp = 3]
  STACK[sp] = 1 ; [3]
  sp++
  STACK[sp] = -1 ; [4]
  sp++
  STACK[sp] = 'AniAttachWeapon' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 118 ; [7]
  sp++
  SYSCALL(1, 0x2f, 118, 2, 'AniAttachWeapon', -1, 1)
  sp -= 5
  STACK[sp] = STACK[fp + 0] ; [3]
  sp++
  STACK[sp] = STACK[fp + 1] ; [4]
  sp++
  STACK[sp] = 'AniBtlWait' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 118 ; [7]
  sp++
  SYSCALL(1, 0x2f, 118, 2, 'AniBtlWait', STACK[fp + 1], STACK[fp + 0])
  sp -= 5
  goto block_43

block_24(0xEFDD0), loc_EFDD0, [sp = 3]
  STACK[sp] = STACK[fp + 2] ; [3]
  sp++
  STACK[sp] = 114 ; [4]
  sp++
  rhs = STACK[--sp]  ; 114
  lhs = STACK[--sp]  ; STACK[fp + 2]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_26 else block_25

block_25(0xEFDE1), loc_EFDE1, [sp = 3]
  STACK[sp] = 1 ; [3]
  sp++
  STACK[sp] = -1 ; [4]
  sp++
  STACK[sp] = 'AniAttachWeapon' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 114 ; [7]
  sp++
  SYSCALL(1, 0x2f, 114, 2, 'AniAttachWeapon', -1, 1)
  sp -= 5
  STACK[sp] = STACK[fp + 0] ; [3]
  sp++
  STACK[sp] = STACK[fp + 1] ; [4]
  sp++
  STACK[sp] = 'AniBtlWait' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 114 ; [7]
  sp++
  SYSCALL(1, 0x2f, 114, 2, 'AniBtlWait', STACK[fp + 1], STACK[fp + 0])
  sp -= 5
  goto block_43

block_26(0xEFE2C), loc_EFE2C, [sp = 3]
  STACK[sp] = STACK[fp + 2] ; [3]
  sp++
  STACK[sp] = 108 ; [4]
  sp++
  rhs = STACK[--sp]  ; 108
  lhs = STACK[--sp]  ; STACK[fp + 2]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_28 else block_27

block_27(0xEFE3D), loc_EFE3D, [sp = 3]
  STACK[sp] = 1 ; [3]
  sp++
  STACK[sp] = -1 ; [4]
  sp++
  STACK[sp] = 'AniAttachWeapon' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 108 ; [7]
  sp++
  SYSCALL(1, 0x2f, 108, 2, 'AniAttachWeapon', -1, 1)
  sp -= 5
  STACK[sp] = STACK[fp + 0] ; [3]
  sp++
  STACK[sp] = STACK[fp + 1] ; [4]
  sp++
  STACK[sp] = 'AniBtlWait' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 108 ; [7]
  sp++
  SYSCALL(1, 0x2f, 108, 2, 'AniBtlWait', STACK[fp + 1], STACK[fp + 0])
  sp -= 5
  goto block_43

block_28(0xEFE88), loc_EFE88, [sp = 3]
  STACK[sp] = STACK[fp + 2] ; [3]
  sp++
  STACK[sp] = 101 ; [4]
  sp++
  rhs = STACK[--sp]  ; 101
  lhs = STACK[--sp]  ; STACK[fp + 2]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_30 else block_29

block_29(0xEFE99), loc_EFE99, [sp = 3]
  STACK[sp] = 1 ; [3]
  sp++
  STACK[sp] = -1 ; [4]
  sp++
  STACK[sp] = 'AniAttachWeapon' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 101 ; [7]
  sp++
  SYSCALL(1, 0x2f, 101, 2, 'AniAttachWeapon', -1, 1)
  sp -= 5
  STACK[sp] = STACK[fp + 0] ; [3]
  sp++
  STACK[sp] = STACK[fp + 1] ; [4]
  sp++
  STACK[sp] = 'AniBtlWait' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 101 ; [7]
  sp++
  SYSCALL(1, 0x2f, 101, 2, 'AniBtlWait', STACK[fp + 1], STACK[fp + 0])
  sp -= 5
  goto block_43

block_30(0xEFEE4), loc_EFEE4, [sp = 3]
  STACK[sp] = STACK[fp + 2] ; [3]
  sp++
  STACK[sp] = 102 ; [4]
  sp++
  rhs = STACK[--sp]  ; 102
  lhs = STACK[--sp]  ; STACK[fp + 2]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_32 else block_31

block_31(0xEFEF5), loc_EFEF5, [sp = 3]
  STACK[sp] = 1 ; [3]
  sp++
  STACK[sp] = -1 ; [4]
  sp++
  STACK[sp] = 'AniAttachWeapon' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 102 ; [7]
  sp++
  SYSCALL(1, 0x2f, 102, 2, 'AniAttachWeapon', -1, 1)
  sp -= 5
  STACK[sp] = STACK[fp + 0] ; [3]
  sp++
  STACK[sp] = STACK[fp + 1] ; [4]
  sp++
  STACK[sp] = 'AniBtlWait' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 102 ; [7]
  sp++
  SYSCALL(1, 0x2f, 102, 2, 'AniBtlWait', STACK[fp + 1], STACK[fp + 0])
  sp -= 5
  goto block_43

block_32(0xEFF40), loc_EFF40, [sp = 3]
  STACK[sp] = STACK[fp + 2] ; [3]
  sp++
  STACK[sp] = 103 ; [4]
  sp++
  rhs = STACK[--sp]  ; 103
  lhs = STACK[--sp]  ; STACK[fp + 2]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_34 else block_33

block_33(0xEFF51), loc_EFF51, [sp = 3]
  STACK[sp] = 1 ; [3]
  sp++
  STACK[sp] = -1 ; [4]
  sp++
  STACK[sp] = 'AniAttachWeapon' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 103 ; [7]
  sp++
  SYSCALL(1, 0x2f, 103, 2, 'AniAttachWeapon', -1, 1)
  sp -= 5
  STACK[sp] = STACK[fp + 0] ; [3]
  sp++
  STACK[sp] = STACK[fp + 1] ; [4]
  sp++
  STACK[sp] = 'AniBtlWait' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 103 ; [7]
  sp++
  SYSCALL(1, 0x2f, 103, 2, 'AniBtlWait', STACK[fp + 1], STACK[fp + 0])
  sp -= 5
  goto block_43

block_34(0xEFF9C), loc_EFF9C, [sp = 3]
  STACK[sp] = STACK[fp + 2] ; [3]
  sp++
  STACK[sp] = 100 ; [4]
  sp++
  rhs = STACK[--sp]  ; 100
  lhs = STACK[--sp]  ; STACK[fp + 2]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_36 else block_35

block_35(0xEFFAD), loc_EFFAD, [sp = 3]
  STACK[sp] = 1 ; [3]
  sp++
  STACK[sp] = -1 ; [4]
  sp++
  STACK[sp] = 'AniAttachWeapon' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 100 ; [7]
  sp++
  SYSCALL(1, 0x2f, 100, 2, 'AniAttachWeapon', -1, 1)
  sp -= 5
  STACK[sp] = STACK[fp + 0] ; [3]
  sp++
  STACK[sp] = STACK[fp + 1] ; [4]
  sp++
  STACK[sp] = 'AniBtlWait' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 100 ; [7]
  sp++
  SYSCALL(1, 0x2f, 100, 2, 'AniBtlWait', STACK[fp + 1], STACK[fp + 0])
  sp -= 5
  goto block_43

block_36(0xEFFF8), loc_EFFF8, [sp = 3]
  STACK[sp] = STACK[fp + 2] ; [3]
  sp++
  STACK[sp] = 137 ; [4]
  sp++
  rhs = STACK[--sp]  ; 137
  lhs = STACK[--sp]  ; STACK[fp + 2]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_38 else block_37

block_37(0xF0009), loc_F0009, [sp = 3]
  STACK[sp] = 1 ; [3]
  sp++
  STACK[sp] = -1 ; [4]
  sp++
  STACK[sp] = 'AniAttachWeapon' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 137 ; [7]
  sp++
  SYSCALL(1, 0x2f, 137, 2, 'AniAttachWeapon', -1, 1)
  sp -= 5
  STACK[sp] = STACK[fp + 0] ; [3]
  sp++
  STACK[sp] = STACK[fp + 1] ; [4]
  sp++
  STACK[sp] = 'AniBtlWait' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 137 ; [7]
  sp++
  SYSCALL(1, 0x2f, 137, 2, 'AniBtlWait', STACK[fp + 1], STACK[fp + 0])
  sp -= 5
  goto block_43

block_38(0xF0054), loc_F0054, [sp = 3]
  STACK[sp] = STACK[fp + 2] ; [3]
  sp++
  STACK[sp] = 138 ; [4]
  sp++
  rhs = STACK[--sp]  ; 138
  lhs = STACK[--sp]  ; STACK[fp + 2]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_40 else block_39

block_39(0xF0065), loc_F0065, [sp = 3]
  STACK[sp] = 1 ; [3]
  sp++
  STACK[sp] = -1 ; [4]
  sp++
  STACK[sp] = 'AniAttachWeapon' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 138 ; [7]
  sp++
  SYSCALL(1, 0x2f, 138, 2, 'AniAttachWeapon', -1, 1)
  sp -= 5
  STACK[sp] = STACK[fp + 0] ; [3]
  sp++
  STACK[sp] = STACK[fp + 1] ; [4]
  sp++
  STACK[sp] = 'AniBtlWait' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 138 ; [7]
  sp++
  SYSCALL(1, 0x2f, 138, 2, 'AniBtlWait', STACK[fp + 1], STACK[fp + 0])
  sp -= 5
  goto block_43

block_40(0xF00B0), loc_F00B0, [sp = 3]
  STACK[sp] = STACK[fp + 2] ; [3]
  sp++
  STACK[sp] = 139 ; [4]
  sp++
  rhs = STACK[--sp]  ; 139
  lhs = STACK[--sp]  ; STACK[fp + 2]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_42 else block_41

block_41(0xF00C1), loc_F00C1, [sp = 3]
  STACK[sp] = 1 ; [3]
  sp++
  STACK[sp] = -1 ; [4]
  sp++
  STACK[sp] = 'AniAttachWeapon' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 139 ; [7]
  sp++
  SYSCALL(1, 0x2f, 139, 2, 'AniAttachWeapon', -1, 1)
  sp -= 5
  STACK[sp] = STACK[fp + 0] ; [3]
  sp++
  STACK[sp] = STACK[fp + 1] ; [4]
  sp++
  STACK[sp] = 'AniBtlWait' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = 139 ; [7]
  sp++
  SYSCALL(1, 0x2f, 139, 2, 'AniBtlWait', STACK[fp + 1], STACK[fp + 0])
  sp -= 5
  goto block_43

block_42(0xF010C), loc_F010C, [sp = 3]
  STACK[sp] = 1 ; [3]
  sp++
  STACK[sp] = -1 ; [4]
  sp++
  STACK[sp] = 'AniAttachWeapon' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = STACK[fp + 2] ; [7]
  sp++
  SYSCALL(1, 0x2f, STACK[fp + 2], 2, 'AniAttachWeapon', -1, 1)
  sp -= 5
  STACK[sp] = STACK[fp + 0] ; [3]
  sp++
  STACK[sp] = STACK[fp + 1] ; [4]
  sp++
  STACK[sp] = 'AniEvBtlWait' ; [5]
  sp++
  STACK[sp] = 2 ; [6]
  sp++
  STACK[sp] = STACK[fp + 2] ; [7]
  sp++
  SYSCALL(1, 0x2f, STACK[fp + 2], 2, 'AniEvBtlWait', STACK[fp + 1], STACK[fp + 0])
  sp -= 5
  goto block_43

block_43(0xF0150), loc_F0150, [sp = 3]
  STACK[sp] = 0x0 ; [3]
  sp++
  REG[0] = STACK[--sp]  ; 0x0
  sp -= 3
  return


; ---------- EV_02_63_02_SELECT_00 ----------
block_0(0x15FBC7), EV_02_63_02_SELECT_00, [sp = 0, fp = 0]
  dbg_line 26456
  STACK[sp] = <func_id> ; [0]
  sp++
  STACK[sp] = <&loc_15FBDF> ; [1]
  sp++
  STACK[sp] = 0 ; [2]
  sp++
  call global_work(0)

block_1(0x15FBDF), loc_15FBDF, [sp = 0]
  STACK[sp] = REG[0] ; [0]
  sp++
  STACK[sp] = -1 ; [1]
  sp++
  rhs = STACK[--sp]  ; -1
  lhs = STACK[--sp]  ; REG[0]
  STACK[sp++] = (lhs == rhs) ? 1 : 0
  if (STACK[--sp] EQ 0) goto block_14 else block_2

block_2(0x15FBED), loc_15FBED, [sp = 0]
  dbg_line 26457
  STACK[sp] = <func_id> ; [0]
  sp++
  STACK[sp] = <&loc_15FC17> ; [1]
  sp++
  STACK[sp] = 28 ; [2]
  sp++
  STACK[sp] = 0 ; [3]
  sp++
  STACK[sp] = 2 ; [4]
  sp++
  STACK[sp] = 0 ; [5]
  sp++
  call menu_create(0, 2, 0, 28)

block_3(0x15FC17), loc_15FC17, [sp = 0]
  dbg_line 26458
  STACK[sp] = <func_id> ; [0]
  sp++
  STACK[sp] = <&loc_15FC3B> ; [1]
  sp++
  STACK[sp] = 0 ; [2]
  sp++
  STACK[sp] = '那好，我也去！' ; [3]
  sp++
  STACK[sp] = 0 ; [4]
  sp++
  call menu_additem(0, '那好，我也去！', 0)

block_4(0x15FC3B), loc_15FC3B, [sp = 0]
  dbg_line 26459
  STACK[sp] = <func_id> ; [0]
  sp++
  STACK[sp] = <&loc_15FC5F> ; [1]
  sp++
  STACK[sp] = 1 ; [2]
  sp++
  STACK[sp] = '这、这家伙也太乱来了……' ; [3]
  sp++
  STACK[sp] = 0 ; [4]
  sp++
  call menu_additem(0, '这、这家伙也太乱来了……', 1)

block_5(0x15FC5F), loc_15FC5F, [sp = 0]
  dbg_line 26461
  STACK[sp] = <func_id> ; [0]
  sp++
  STACK[sp] = <&loc_15FC8F> ; [1]
  sp++
  STACK[sp] = '' ; [2]
  sp++
  STACK[sp] = 0 ; [3]
  sp++
  STACK[sp] = -1 ; [4]
  sp++
  STACK[sp] = -1 ; [5]
  sp++
  STACK[sp] = 0 ; [6]
  sp++
  call menu_open(0, -1, -1, 0, '')

block_6(0x15FC8F), loc_15FC8F, [sp = 0]
  dbg_line 26464
  STACK[sp] = 0x0 ; [0]
  sp++
  STACK[sp] = <func_id> ; [1]
  sp++
  STACK[sp] = <&loc_15FCAD> ; [2]
  sp++
  STACK[sp] = 0 ; [3]
  sp++
  call menu_wait(0)

block_7(0x15FCAD), loc_15FCAD, [sp = 1]
  STACK[sp] = REG[0] ; [1]
  sp++
  STACK[0] = STACK[--sp] ; REG[0]

  dbg_line 26465
  STACK[sp] = <func_id> ; [1]
  sp++
  STACK[sp] = <&loc_15FCCC> ; [2]
  sp++
  STACK[sp] = 0 ; [3]
  sp++
  call menu_close(0)

block_8(0x15FCCC), loc_15FCCC, [sp = 1]
  dbg_line 26467
  STACK[sp] = <func_id> ; [1]
  sp++
  STACK[sp] = <&loc_15FCE4> ; [2]
  sp++
  STACK[sp] = 0 ; [3]
  sp++
  call menu_is_canceled(0)

block_9(0x15FCE4), loc_15FCE4, [sp = 1]
  STACK[sp] = REG[0] ; [1]
  sp++
  if (STACK[--sp] EQ 0) goto block_12 else block_10

block_10(0x15FCEB), loc_15FCEB, [sp = 1]
  dbg_line 26468
  STACK[sp] = <func_id> ; [1]
  sp++
  STACK[sp] = <&loc_15FD09> ; [2]
  sp++
  STACK[sp] = 1 ; [3]
  sp++
  STACK[sp] = 0 ; [4]
  sp++
  call set_global_work(0, 1)

block_11(0x15FD09), loc_15FD09, [sp = 1]
  goto block_13

block_12(0x15FD0E), loc_15FD0E, [sp = 1]
  dbg_line 26470
  STACK[sp] = <func_id> ; [1]
  sp++
  STACK[sp] = <&loc_15FD2B> ; [2]
  sp++
  STACK[sp] = STACK[sp - 3<0>] ; [3]
  sp++
  STACK[sp] = 0 ; [4]
  sp++
  call set_global_work(0, STACK[sp - 3<0>])

block_13(0x15FD2B), loc_15FD2B, [sp = 1]
  sp--
  goto block_14

block_14(0x15FD2D), loc_15FD2D, [sp = 0]
  dbg_line 26473
  STACK[sp] = 0x0 ; [0]
  sp++
  REG[0] = STACK[--sp]  ; 0x0
  return


