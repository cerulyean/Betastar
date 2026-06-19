import sc2reader

replay_path = r"D:\betastar\BREATH_3\rfc_10k_2\27122786_480_5088.5_Hupsaiya.SC2Replay"
replay = sc2reader.load_replay(replay_path, load_level=1)
print(replay.map_name)
