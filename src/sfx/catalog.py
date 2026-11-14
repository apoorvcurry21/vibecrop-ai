"""The sound effects the library is built from. Text only: no audio is committed (the sounds are copyrighted and the
repo is public). `scripts/build_sfx_library.py` downloads each entry once into the local sfx folder.

An entry is (id, display name, aliases, YouTube search query). Aliases are other ways a user might ask for it
(English and Hinglish). Edit this list freely, then run the build script again: it only fetches what is missing.
"""
from __future__ import annotations

# (id, name, aliases, search query)
_ROWS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    # ---- meme classics
    ("fah", "Fah", ("fahhh", "faaah", "fa", "fah sound"), "fahhh sound effect meme"),
    ("vine_boom", "Vine boom", ("boom", "vine", "dramatic boom"), "vine boom sound effect"),
    ("bruh", "Bruh", ("bro", "bruh sound"), "bruh sound effect"),
    ("airhorn", "Airhorn", ("mlg airhorn", "air horn", "horn"), "mlg airhorn sound effect"),
    ("sad_violin", "Sad violin", ("violin", "sad music", "dukhi"), "sad violin meme sound effect"),
    ("record_scratch", "Record scratch", ("scratch", "freeze frame", "wait what"), "record scratch sound effect"),
    ("crickets", "Crickets", ("awkward silence", "cricket sound", "silence"), "crickets chirping sound effect meme"),
    ("dun_dun_dun", "Dun dun dun", ("dramatic", "dun dun", "suspense"), "dun dun dun sound effect"),
    ("sheesh", "Sheesh", ("sheeesh",), "sheesh sound effect"),
    ("bonk", "Bonk", ("go to horny jail", "bonk sound"), "bonk sound effect meme"),
    ("emotional_damage", "Emotional damage", ("damage", "emotional"), "emotional damage sound effect"),
    ("among_us_sus", "Among Us sus", ("sus", "among us", "impostor", "suspicious"), "among us sus sound effect"),
    ("taco_bell_bong", "Taco Bell bong", ("bong", "taco bell"), "taco bell bong sound effect"),
    ("windows_error", "Windows error", ("error", "windows xp error", "critical error"), "windows xp error sound effect"),
    ("oh_no_no_no", "Oh no no no", ("no no no", "oh no", "nooo"), "oh no no no tiktok sound effect"),
    ("gta_wasted", "GTA wasted", ("wasted", "you died", "gta"), "gta wasted sound effect"),
    ("mission_passed", "Mission passed", ("respect", "mission complete", "respect plus"), "gta mission passed respect sound effect"),
    ("metal_gear_alert", "Metal Gear alert", ("alert", "caught", "exclamation", "spotted"), "metal gear solid alert sound effect"),
    ("anime_wow", "Anime wow", ("wow anime", "sugoi", "anime sound"), "anime wow sound effect"),
    ("roblox_oof", "Roblox oof", ("oof", "roblox"), "roblox oof sound effect"),
    ("owen_wilson_wow", "Owen Wilson wow", ("wow", "mmm wow", "owen wilson"), "owen wilson wow sound effect"),
    ("sad_trombone", "Sad trombone", ("fail", "wah wah", "womp womp", "trombone"), "sad trombone fail sound effect"),
    ("rimshot", "Rimshot", ("ba dum tss", "badum tss", "joke drum", "drum"), "ba dum tss rimshot sound effect"),
    ("gasp", "Gasp", ("shock", "gasp sound", "audience gasp"), "audience gasp sound effect"),
    ("boing", "Boing", ("spring", "bounce", "cartoon boing"), "cartoon boing sound effect"),
    ("huh_cat", "Huh cat", ("huh", "confused", "kya"), "huh cat meme sound effect"),
    # ---- Indian memes (Hinglish)
    ("bhai_kya_kar_raha_hai_tu", "Bhai kya kar raha hai tu", ("bhai kya kar raha hai", "kya kar raha hai tu", "bhai kya kar rha hai tu"),
     "bhai kya kar raha hai tu meme sound effect"),
    ("arey_bhai_bhai_bhai", "Arey bhai bhai bhai", ("arey bhai", "are bhai bhai bhai"), "arey bhai bhai bhai meme sound effect"),
    ("pagal_hai_kya", "Pagal hai kya", ("pagal hai", "pagal"), "pagal hai kya meme sound effect"),
    ("kya_baat_hai", "Kya baat hai", ("kya baat", "waah"), "kya baat hai meme sound effect"),
    ("mauka_mauka", "Mauka mauka", ("mauka", "india pakistan mauka"), "mauka mauka star sports world cup ad india pakistan"),
    ("jethalal_bapuji", "Jethalal bapuji", ("bapuji", "jethalal", "tmkoc"), "jethalal bapuji meme sound effect"),
    ("ye_kya_ho_raha_hai", "Ye kya ho raha hai", ("baburao", "yeh kya ho raha hai", "hera pheri"), "baburao ye kya ho raha hai sound effect"),
    ("abe_saale", "Abe saale", ("abey saale", "abe sale"), "abe saale meme sound effect"),
    ("dhinchak", "Dhinchak", ("dhinchak pooja", "dhinchak sound"), "dhinchak pooja sound effect meme"),
    ("oh_my_god_shaka_laka", "Oh my god (Shaka Laka)", ("omg", "oh my god", "shaka laka"), "oh my god shaka laka boom sound effect"),
    ("hey_ram", "Hey Ram", ("he ram", "hey ram sound"), "hey ram meme sound effect"),
    # ---- utility
    ("whoosh", "Whoosh", ("swoosh", "woosh", "transition"), "whoosh transition sound effect"),
    ("pop", "Pop", ("pop sound", "bubble pop"), "pop sound effect"),
    ("ding", "Ding", ("notification", "bell ding", "correct"), "ding notification sound effect"),
    ("cash_register", "Cash register", ("cha ching", "money", "kaching"), "cash register cha ching sound effect"),
    ("camera_shutter", "Camera shutter", ("camera", "photo click", "click"), "camera shutter sound effect"),
    ("applause", "Applause", ("clap", "clapping", "taliyan"), "applause sound effect short"),
    ("laugh_track", "Laugh track", ("laugh", "sitcom laugh", "hasi"), "sitcom laugh track sound effect"),
    ("glass_break", "Glass break", ("glass", "shatter", "break"), "glass breaking sound effect"),
    ("door_knock", "Door knock", ("knock", "knock knock"), "door knock sound effect"),
    ("explosion", "Explosion", ("blast", "bomb", "boom explosion"), "explosion sound effect short"),
    ("record_stop", "Record stop", ("tape stop", "stop", "power down"), "tape stop sound effect"),
    ("swoosh_transition", "Swoosh transition", ("swish", "whoosh fast", "fast swoosh"), "fast swoosh transition sound effect"),
    ("heartbeat", "Heartbeat", ("heart beat", "dhak dhak"), "heartbeat sound effect"),
    ("typing", "Typing", ("keyboard", "keyboard typing"), "keyboard typing sound effect short"),
    ("tada", "Ta-da", ("tada", "ta da", "success", "fanfare"), "ta da fanfare sound effect"),
    ("level_up", "Level up", ("power up", "coin", "mario coin"), "level up sound effect"),
)

CATALOG: tuple[dict, ...] = tuple(
    {"id": i, "name": n, "aliases": list(a), "query": q} for i, n, a, q in _ROWS
)
