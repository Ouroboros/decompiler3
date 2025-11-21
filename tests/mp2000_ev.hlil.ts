function CHR_SETARM(arg3, arg1, arg2) {
    switch (arg3) {
      case 2:
            syscall(1, 47, 2, 2, "AniAttachWeapon", -1, 1.0);
            syscall(1, 47, 2, 2, "AniBtlWait", arg2, arg1);
        break;
      case 1:
            syscall(1, 47, 1, 2, "AniAttachWeapon", -1, 1.0);
            syscall(1, 47, 1, 2, "AniBtlWait", arg2, arg1);
        break;
      case 0:
            syscall(1, 47, 0, 2, "AniAttachWeapon", -1, 1.0);
            syscall(1, 47, 0, 2, "AniBtlWait", arg2, arg1);
        break;
      default:
            switch (arg3) {
              case 5:
                    syscall(1, 47, 5, 2, "AniAttachWeapon", -1, 1.0);
                    syscall(1, 47, 5, 2, "AniBtlWait", arg2, arg1);
                break;
              case 4:
                    syscall(1, 47, 4, 2, "AniAttachWeapon", -1, 1.0);
                    syscall(1, 47, 4, 2, "AniBtlWait", arg2, arg1);
                break;
              case 3:
                    syscall(1, 47, 3, 2, "AniAttachWeapon", -1, 1.0);
                    syscall(1, 47, 3, 2, "AniBtlWait", arg2, arg1);
                break;
              default:
                    switch (arg3) {
                      case 106:
                            syscall(1, 47, 106, 2, "AniAttachWeapon", -1, 1.0);
                            syscall(1, 47, 106, 2, "AniBtlWait", arg2, arg1);
                        break;
                      case 7:
                            syscall(1, 47, 7, 2, "AniAttachWeapon", -1, 1.0);
                            syscall(1, 47, 7, 2, "AniBtlWait", arg2, arg1);
                        break;
                      case 6:
                            syscall(1, 47, 6, 2, "AniAttachWeapon", -1, 1.0);
                            syscall(1, 47, 6, 2, "AniBtlWait", arg2, arg1);
                        break;
                      default:
                            switch (arg3) {
                              case 118:
                                    syscall(1, 47, 118, 2, "AniAttachWeapon", -1, 1.0);
                                    syscall(1, 47, 118, 2, "AniBtlWait", arg2, arg1);
                                break;
                              case 104:
                                    syscall(1, 47, 104, 2, "AniAttachWeapon", -1, 1.0);
                                    syscall(1, 47, 104, 2, "AniBtlWait", arg2, arg1);
                                break;
                              case 105:
                                    syscall(1, 47, 105, 2, "AniAttachWeapon", -1, 1.0);
                                    syscall(1, 47, 105, 2, "AniBtlWait", arg2, arg1);
                                break;
                              default:
                                    switch (arg3) {
                                      case 101:
                                            syscall(1, 47, 101, 2, "AniAttachWeapon", -1, 1.0);
                                            syscall(1, 47, 101, 2, "AniBtlWait", arg2, arg1);
                                        break;
                                      case 108:
                                            syscall(1, 47, 108, 2, "AniAttachWeapon", -1, 1.0);
                                            syscall(1, 47, 108, 2, "AniBtlWait", arg2, arg1);
                                        break;
                                      case 114:
                                            syscall(1, 47, 114, 2, "AniAttachWeapon", -1, 1.0);
                                            syscall(1, 47, 114, 2, "AniBtlWait", arg2, arg1);
                                        break;
                                      default:
                                            switch (arg3) {
                                              case 100:
                                                    syscall(1, 47, 100, 2, "AniAttachWeapon", -1, 1.0);
                                                    syscall(1, 47, 100, 2, "AniBtlWait", arg2, arg1);
                                                break;
                                              case 103:
                                                    syscall(1, 47, 103, 2, "AniAttachWeapon", -1, 1.0);
                                                    syscall(1, 47, 103, 2, "AniBtlWait", arg2, arg1);
                                                break;
                                              case 102:
                                                    syscall(1, 47, 102, 2, "AniAttachWeapon", -1, 1.0);
                                                    syscall(1, 47, 102, 2, "AniBtlWait", arg2, arg1);
                                                break;
                                              default:
                                                    switch (arg3) {
                                                      case 139:
                                                            syscall(1, 47, 139, 2, "AniAttachWeapon", -1, 1.0);
                                                            syscall(1, 47, 139, 2, "AniBtlWait", arg2, arg1);
                                                        break;
                                                      case 138:
                                                            syscall(1, 47, 138, 2, "AniAttachWeapon", -1, 1.0);
                                                            syscall(1, 47, 138, 2, "AniBtlWait", arg2, arg1);
                                                        break;
                                                      case 137:
                                                            syscall(1, 47, 137, 2, "AniAttachWeapon", -1, 1.0);
                                                            syscall(1, 47, 137, 2, "AniBtlWait", arg2, arg1);
                                                        break;
                                                      default:
                                                            syscall(1, 47, arg3, 2, "AniAttachWeapon", -1, 1.0);
                                                            syscall(1, 47, arg3, 2, "AniEvBtlWait", arg2, arg1);
                                                        break;
                                                    }
                                                break;
                                            }
                                        break;
                                    }
                                break;
                            }
                        break;
                    }
                break;
            }
        break;
    }
    return 0;
}

function EV_02_63_02_SELECT_00() {
    let var_s0;

    // line(26456)
    if (global_work(0) == -1) {
        // line(26457)
        menu_create(0, 2, 0, 28);
        // line(26458)
        menu_additem(0, "那好，我也去！", 0);
        // line(26459)
        menu_additem(0, "这、这家伙也太乱来了……", 1);
        // line(26461)
        menu_open(0, -1, -1, 0, "");
        // line(26464)
        var_s0 = menu_wait(0);
        // line(26465)
        menu_close(0);
        // line(26467)
        if (menu_is_canceled(0) == 0) {
            // line(26470)
            set_global_work(0, var_s0);
            // line(26473)
            return 0;
        } else {
            // line(26468)
            set_global_work(0, 1);
            // line(26473)
            return 0;
        }
    }
    // line(26473)
    return 0;
}

