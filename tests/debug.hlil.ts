function Dummy_m4000_talk0() {
    let var_s2;
    let var_s3;
    let var_s4;
    let var_s5;
    let var_s0;
    let var_s1;
    let var_s6;
    let var_s7;
    let var_s8;
    let var_s9;

    // line(10957)
    TALK_BEGIN(0, 4.0);
    // line(10960)
    menu_create(0, 0, 0, 24);
    // line(10963)
    menu_additem(0, "中間地点①", 0);
    // line(10965)
    menu_open(0, -1, -1, 1);
    // line(10967)
    var_s0 = menu_wait(0);
    // line(10968)
    menu_close(0);
    // line(10970)
    if (var_s0 < 0) {
    } else {
        // line(10972)
        var_s6 = 0.5;
        fade_out(0.5, 0, 1.0, 0);
        // line(10973)
        fade_wait(0);
        // line(10975)
        if (var_s0 != 0) {
            // line(10979)
            if (var_s0 != 1) {
                // line(10983)
                if (var_s0 != 2) {
                } else {
                    // line(10984)
                    chr_set_pos(65000, -5.976, -0.283, -157.72, 112.963);
                    // line(10985)
                    var_s6 = 0.0;
                    var_s7 = 0.0;
                    var_s8 = 0.0;
                    var_s9 = 65000;
                    camera_rotate_chr(65000, 0.0, 0.0, 0.0, 0, 3, -1);
                }
            } else {
                // line(10980)
                chr_set_pos(65000, -434.334, 5.358, 42.086, 355.173);
                // line(10981)
                var_s6 = 0.0;
                var_s7 = 0.0;
                var_s8 = 0.0;
                var_s9 = 65000;
                camera_rotate_chr(65000, 0.0, 0.0, 0.0, 0, 3, -1);
            }
        } else {
            // line(10976)
            chr_set_pos(65000, -259.911, -0.0, 203.559, 172.54);
            // line(10977)
            var_s6 = 0.0;
            var_s7 = 0.0;
            var_s8 = 0.0;
            var_s9 = 65000;
            camera_rotate_chr(65000, 0.0, 0.0, 0.0, 0, 3, -1);
        }
    }
    // line(10989)
    TALK_END();
    // line(10991)
    if (var_s0 < 0) {
    } else {
        // line(10992)
        fade_in(0.5, 0, 0.0, 0);
    }
    // line(10995)
    return 0;
}

