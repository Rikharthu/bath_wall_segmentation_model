# TODO: filter scenes
# TODO: compare with LIST_SCENES in WallSegmentation
#   https://github.com/bjekic/WallSegmentation/blob/main/utils/constants.py
#   What they have extra?
#   What we have extra?
#   Number of samples (their, mine, their original since dataset was updated)
scene_whitelist = [
    'art_gallery',
    'bathroom',
    'bedroom',
    'conference_room',
    'corridor',
    'day_care_center',
    'kitchen',
    'library',
    'living_room',
    'locker_room',
    'office',
    'poolroom',
    'elevator',
    'alcove',
    'amusement_arcade',
    'amusement_park',
    'archive',
    'art_studio',
    'artists_loft',
    'attic',
    'banquet_hall',
    'bar',
    'basement',
    'beauty_salon',
    'bedchamber',
    'berth',
    'biology_laboratory',
    'bookstore',
    'cafeteria',
    'childs_room',
    'classroom',
    'clean_room',
    'coffee_shop',
    'computer_room',
    'dining_hall',
    'dining_room',
    'dorm_room',
    'dressing_room',
    'entrance_hall',
    'home_office',
    'home_theater',
    'hotel_room',
    'ice_cream_parlor',
    'jail_cell',
    'japanese_garden',
    'kindergarden_classroom',
    'lecture_room',
    'legislative_chamber',
    'lobby',
    'martial_arts_gym',
    'mezzanine',
    'music_studio',
    'nursery',
    'nursing_home',
    'office_cubicles',
    'operating_room',
    'pantry',
    'pet_shop',
    'pharmacy',
    'playroom',
    'reception',
    'recreation_room',
    'repair_shop',
    'restaurant',
    'restaurant_kitchen',
    'science_museum',
    'server_room',
    'shower',
    'staircase', # Could be filtered
    'television_room',
    'television_studio',
    'throne_room',
    'toyshop',
    'utility_room',
    'veterinarians_office',
    'waiting_room',
    'wet_bar',
    'atrium',
    'basketball_court',
    'bow_window',
    'doorway',
    'movie_theater',
]
# full scene names
full_scene_whitelist = [
    'swimming_pool/indoor'
]
individual_samples_whitelist = [
    'ADE_train_00012308.jpg',
    'ADE_train_00012302.jpg',
    'ADE_train_00012299.jpg',
    'ADE_train_00012294.jpg',
    'ADE_train_00012293.jpg',
    'ADE_train_00014248.jpg',
    'ADE_train_00014249.jpg',
    'ADE_train_00014252.jpg',
    'ADE_train_00022651.jpg',
    'ADE_train_00023134.jpg',
    'ADE_train_00023339.jpg',
    'ADE_train_00023723.jpg',
    'ADE_train_00024896.jpg',
    'ADE_train_00024895.jpg',
    'ADE_train_00024988.jpg',
    'ADE_train_00025006.jpg',
    'ADE_train_00025003.jpg',
    'ADE_train_00025155.jpg',
    'ADE_train_00025174.jpg',
]
scene_whitelist_maybe = [
    'booth',
    'joss_house',
    'supermarket',
    'swimming_pool',  # /swimming_pool/indoor only
    'aquarium',
    'arcade',
    'florist_shop',
]
