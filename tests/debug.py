def Dummy_m4000_talk0():
    DEBUG_SET_LINENO(10957)
    PUSH_CURRENT_FUNC_ID()
    PUSH_RET_ADDR(loc_8B995)
    PUSH_FLOAT(4.0)
    PUSH_INT(0)
    CALL(TALK_BEGIN)

    label('loc_8B995')
    
    DEBUG_SET_LINENO(10960)
    PUSH_CURRENT_FUNC_ID()
    PUSH_RET_ADDR(loc_8B9BF)
    PUSH_INT(24)
    PUSH_INT(0)
    PUSH_INT(0)
    PUSH_INT(0)
    CALL(menu_create)

    label('loc_8B9BF')
    
    DEBUG_SET_LINENO(10963)
    PUSH_CURRENT_FUNC_ID()
    PUSH_RET_ADDR(loc_8B9E3)
    PUSH_INT(0)
    PUSH_STR("中間地点①")
    PUSH_INT(0)
    CALL(menu_additem)

    label('loc_8B9E3')
    
    DEBUG_SET_LINENO(10965)
    PUSH_CURRENT_FUNC_ID()
    PUSH_RET_ADDR(loc_8BA0D)
    PUSH_INT(1)
    PUSH_INT(-1)
    PUSH_INT(-1)
    PUSH_INT(0)
    CALL(menu_open)

    label('loc_8BA0D')
    
    DEBUG_SET_LINENO(10967)
    PUSH_RAW(RawInt(0x00000000))
    PUSH_CURRENT_FUNC_ID()
    PUSH_RET_ADDR(loc_8BA2B)
    PUSH_INT(0)
    CALL(menu_wait)

    label('loc_8BA2B')
    
    GET_REG(0)
    POP_TO(-4)
    DEBUG_SET_LINENO(10968)
    PUSH_CURRENT_FUNC_ID()
    PUSH_RET_ADDR(loc_8BA4A)
    PUSH_INT(0)
    CALL(menu_close)

    label('loc_8BA4A')
    
    DEBUG_SET_LINENO(10970)
    LOAD_STACK(-4)
    PUSH_INT(0)
    GE()
    POP_JMP_ZERO(loc_8BC2A)

    label('loc_8BA5E')
    
    DEBUG_SET_LINENO(10972)
    PUSH_CURRENT_FUNC_ID()
    PUSH_RET_ADDR(loc_8BA88)
    PUSH_INT(0)
    PUSH_FLOAT(1.0)
    PUSH_INT(0)
    PUSH_FLOAT(0.5)
    CALL(fade_out)

    label('loc_8BA88')
    
    DEBUG_SET_LINENO(10973)
    PUSH_CURRENT_FUNC_ID()
    PUSH_RET_ADDR(loc_8BAA0)
    PUSH_INT(0)
    CALL(fade_wait)

    label('loc_8BAA0')
    
    DEBUG_SET_LINENO(10975)
    LOAD_STACK(-4)
    PUSH_INT(0)
    EQ()
    POP_JMP_ZERO(loc_8BB25)

    label('loc_8BAB4')
    
    DEBUG_SET_LINENO(10976)
    PUSH_CURRENT_FUNC_ID()
    PUSH_RET_ADDR(loc_8BAE4)
    PUSH_FLOAT(172.53997802734375)
    PUSH_FLOAT(203.5589599609375)
    PUSH_FLOAT(-0.0)
    PUSH_FLOAT(-259.9110107421875)
    PUSH_INT(65000)
    CALL(chr_set_pos)

    label('loc_8BAE4')
    
    DEBUG_SET_LINENO(10977)
    PUSH_CURRENT_FUNC_ID()
    PUSH_RET_ADDR(loc_8BB20)
    PUSH_INT(-1)
    PUSH_INT(3)
    PUSH_INT(0)
    PUSH_FLOAT(0.0)
    PUSH_FLOAT(0.0)
    PUSH_FLOAT(0.0)
    PUSH_INT(65000)
    CALL(camera_rotate_chr)

    label('loc_8BB20')
    
    JMP(loc_8BC2A)

    label('loc_8BB25')
    
    DEBUG_SET_LINENO(10979)
    LOAD_STACK(-4)
    PUSH_INT(1)
    EQ()
    POP_JMP_ZERO(loc_8BBAA)

    label('loc_8BB39')
    
    DEBUG_SET_LINENO(10980)
    PUSH_CURRENT_FUNC_ID()
    PUSH_RET_ADDR(loc_8BB69)
    PUSH_FLOAT(355.1729736328125)
    PUSH_FLOAT(42.08599853515625)
    PUSH_FLOAT(5.357999801635742)
    PUSH_FLOAT(-434.333984375)
    PUSH_INT(65000)
    CALL(chr_set_pos)

    label('loc_8BB69')
    
    DEBUG_SET_LINENO(10981)
    PUSH_CURRENT_FUNC_ID()
    PUSH_RET_ADDR(loc_8BBA5)
    PUSH_INT(-1)
    PUSH_INT(3)
    PUSH_INT(0)
    PUSH_FLOAT(0.0)
    PUSH_FLOAT(0.0)
    PUSH_FLOAT(0.0)
    PUSH_INT(65000)
    CALL(camera_rotate_chr)

    label('loc_8BBA5')
    
    JMP(loc_8BC2A)

    label('loc_8BBAA')
    
    DEBUG_SET_LINENO(10983)
    LOAD_STACK(-4)
    PUSH_INT(2)
    EQ()
    POP_JMP_ZERO(loc_8BC2A)

    label('loc_8BBBE')
    
    DEBUG_SET_LINENO(10984)
    PUSH_CURRENT_FUNC_ID()
    PUSH_RET_ADDR(loc_8BBEE)
    PUSH_FLOAT(112.96298217773438)
    PUSH_FLOAT(-157.719970703125)
    PUSH_FLOAT(-0.28299999237060547)
    PUSH_FLOAT(-5.97599983215332)
    PUSH_INT(65000)
    CALL(chr_set_pos)

    label('loc_8BBEE')
    
    DEBUG_SET_LINENO(10985)
    PUSH_CURRENT_FUNC_ID()
    PUSH_RET_ADDR(loc_8BC2A)
    PUSH_INT(-1)
    PUSH_INT(3)
    PUSH_INT(0)
    PUSH_FLOAT(0.0)
    PUSH_FLOAT(0.0)
    PUSH_FLOAT(0.0)
    PUSH_INT(65000)
    CALL(camera_rotate_chr)

    label('loc_8BC2A')
    
    DEBUG_SET_LINENO(10989)
    PUSH_CURRENT_FUNC_ID()
    PUSH_RET_ADDR(loc_8BC3C)
    CALL(TALK_END)

    label('loc_8BC3C')
    
    DEBUG_SET_LINENO(10991)
    LOAD_STACK(-4)
    PUSH_INT(0)
    GE()
    POP_JMP_ZERO(loc_8BC7A)

    label('loc_8BC50')
    
    DEBUG_SET_LINENO(10992)
    PUSH_CURRENT_FUNC_ID()
    PUSH_RET_ADDR(loc_8BC7A)
    PUSH_INT(0)
    PUSH_FLOAT(0.0)
    PUSH_INT(0)
    PUSH_FLOAT(0.5)
    CALL(fade_in)

    label('loc_8BC7A')
    
    DEBUG_SET_LINENO(10995)
    PUSH_RAW(RawInt(0x00000000))
    SET_REG(0)
    POP(4)
    RETURN()

