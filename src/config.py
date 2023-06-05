# RESUME_FROM_CHECKPOINT_PATH = None # Do not resume, start from scratch
RESUME_FROM_CHECKPOINT_PATH = '/home/ricardsku/Development/Bath/bath_wall_segmentation_model/notebooks/epoch=159-train_loss=0.0559-val_loss=0.0692-train_dataset_iou=0.8302-val_dataset_iou=0.8128.ckpt'

# DATA_ROOT = "../dataset/ADE20K_2021_17_01"
DATA_ROOT = '/home/ricardsku/Development/ADE20K_2021_17_01'

# TODO: try training DeepLabV3Plus with Jaccard + CrossEntropy / Jaccard + Focal losses
# TODO: try mobileones5 + UNet
ARCHITECTURE = 'DeepLabV3Plus'
# ARCHITECTURE = 'DeepLabV3'
# ARCHITECTURE = 'PSPNet'
# ARCHITECTURE = 'UNet'
# ARCHITECTURE = 'PAN' # TODO
# ENCODER = 'mobileone_s0'
# ENCODER = 'mobileone_s1'
# ENCODER = 'mobileone_s2'
ENCODER = 'mobileone_s3'
# ENCODER = 'mobileone_s4'
# ENCODER = 'mobileone_s5'

# TODO: try this parameter, it defaults to 5 for DeepLabV3+
# ENCODER_DEPTH = 3
ENCODER_DEPTH = 5

ADE20K_WALL_CLASS_IDX = 2977
# 0 is reserved for background
ADE20K_WALL_CLASS_ID = ADE20K_WALL_CLASS_IDX + 1

# TODO: try different image sizes
#   For instance, car segmentation used (320, 320)
# TODO: see size used by WallSegmentation
# INPUT_IMAGE_SIZE = (512, 512)
# TODO: try this with UNet or PSPNet
INPUT_IMAGE_SIZE = (800, 800)
# INPUT_IMAGE_SIZE = (768, 768)
# INPUT_IMAGE_SIZE = (704, 704)
# INPUT_IMAGE_SIZE = (320, 320)

# LEARNING_RATE = 1e-4
LEARNING_RATE = 1e-3

BATCH_SIZE = 4
# BATCH_SIZE = 8
# BATCH_SIZE = 16
# BATCH_SIZE = 32

# Used to limit dataset for debug.
# Set to 'None' to train on whole dataset
TRAIN_SIZE = None
# TRAIN_SIZE = 128

MAX_EPOCHS = 200
# MAX_EPOCHS = 5

# EARLYSTOP_PATIENCE = 5
# EARLYSTOP_PATIENCE = MAX_EPOCHS
# Normally we would set it to small number, such as 3 or 5,
# but we make it larger to get nice-looking graphs depipcting overfitting or underfitting, if any.
# EARLYSTOP_PATIENCE = 10
EARLYSTOP_PATIENCE = 30

FREEZE_ENCODER = False

WALL_SCENES = {'/airlock', '/alcove', '/amusement_arcade', '/anechoic_chamber', '/arcade', '/archive', '/armory',
               '/army_base/n_soldiers_eating_or_drinking_in_the_base', '/army_base/n_soldiers_practicing_shooting',
               '/art_gallery', '/art_school', '/art_studio', '/art_studio/n_artists_explaining_their_piece_of_art',
               '/artists_loft', '/assembly_line', '/atrium/home', '/attic', '/auditorium', '/auto_mechanics/indoor',
               '/auto_showroom', '/backstage', '/badminton_court/indoor', '/bakery/kitchen', '/bakery/shop',
               '/balcony/interior', '/ball_pit', '/ball_pit/n_people_or_kids_leaving_the_ball_pit', '/ballroom',
               '/ballroom/n_dancers_dancing_on_the_dance_floor', '/ballroom/n_dancers_entering_the_dance_floor',
               '/bank/indoor', '/bank_vault', '/banquet_hall', '/baptistry/indoor', '/bar',
               '/bar/n_barmen_serving_food_or_drinks_to_m_clients', '/barbershop', '/barrack', '/basement',
               '/basketball_court/indoor', '/basketball_court_indoor/a_manager_giving_instructions_to_the_team',
               '/bathhouse', '/bathroom', '/beauty_salon', '/bedchamber', '/bedroom',
               '/bedroom/n_people_sleeping_in_the_bed', '/beer_hall', '/belfry', '/berth', '/berth_deck', '/bindery',
               '/biology_laboratory', '/bistro/indoor', '/boat_deck', '/bomb_shelter/indoor', '/bookbindery',
               '/bookstore', '/booth/indoor', '/bow_window/indoor', '/bowling_alley', '/boxing_ring', '/breakroom',
               '/brewery/indoor', '/brickyard/indoor', '/burial_chamber', '/butchers_shop', '/cabin/indoor',
               '/cafeteria', '/call_center', '/candy_store', '/canteen', '/cardroom', '/cargo_container_interior',
               '/cargo_deck/airplane', '/casino/indoor', '/catacomb', '/catwalk', '/chapel', '/checkout_counter',
               '/cheese_factory', '/chemistry_lab', '/chicken_coop/indoor', '/chicken_coop/outdoor',
               '/chicken_farm/outdoor', '/childs_room', '/childs_room/n_kids_reading_a_book_to_m_children',
               '/choir_loft/exterior', '/choir_loft/interior', '/church/indoor', '/classroom',
               '/classroom/n_students_taking_notes_on_a_notebook_or_a_computer', '/clean_room', '/cloakroom/room',
               '/clock_tower/indoor', '/cloister/indoor', '/closet', '/coffee_shop',
               '/coffee_shop/n_people_reading_a_book_or_the_news',
               '/coffee_shop/n_people_talking_or_eating_or_drinking_in_the_bar_or_a_table', '/computer_room',
               '/conference_center', '/conference_center/n_speakers_making_a_presentation', '/conference_hall',
               '/conference_room', '/confessional', '/control_room', '/convenience_store/indoor', '/corridor',
               '/courtroom', '/cubicle/library', '/cubicle/office', '/cybercafe', '/dance_school', '/darkroom',
               '/day_care_center', '/delicatessen', '/dentists_office', '/departure_lounge', '/dinette/home',
               '/dinette/vehicle', '/dining_car', '/dining_hall', '/dining_room', '/distillery', '/doorway/indoor',
               '/doorway/outdoor', '/dorm_room',
               '/dorm_room/n_people_studying_in_the_bed_or_desk_by_reading_a_book_or_being_on_the_computer',
               '/dress_shop', '/dressing_room', '/drugstore', '/dugout', '/editing_room', '/elevator/door',
               '/elevator/exterior', '/elevator/freight_elevator', '/elevator/interior', '/elevator_lobby',
               '/elevator_shaft', '/engine_room', '/entrance_hall', '/escalator/indoor', '/fastfood_restaurant',
               '/ferryboat/indoor', '/firing_range/indoor', '/fishmarket', '/fitting_room/exterior',
               '/fitting_room/interior', '/flea_market/indoor', '/florist_shop/indoor', '/food_court',
               '/foundry/indoor', '/funeral_chapel', '/funeral_home', '/furnace_room', '/galley', '/game_room',
               '/garage/indoor', '/general_store/indoor', '/geodesic_dome/indoor', '/gift_shop', '/gun_store',
               '/gymnasium/indoor', '/gymnasium_indoor/n_instructors_giving_a_group_class', '/hallway',
               '/hallway/n_people_talking', '/hangar/indoor', '/hat_shop', '/hayloft', '/hearth', '/home_office',
               '/home_office/a_person_staring_at_the_computer', '/home_theater', '/hospital_room',
               '/hospital_room/n_doctors_taking_notes_while_checking_m_patients',
               '/hospital_room/n_doctors_talking_to_m_familiy_members', '/hot_tub/indoor', '/hotel_breakfast_area',
               '/hotel_room', '/hotel_room/n_cleaning-staff_cleaning_the_room', '/hunting_lodge/indoor',
               '/ice_cream_parlor',
               '/ice_cream_parlor/n_customers_looking_or_pointing_at_ice-cream_flavors__in_the_showcase', '/inn/indoor',
               '/jacuzzi/indoor', '/jail_cell', '/jewelry_shop', '/jewelry_shop/n_employees_crafting_or_fixing_a_jewel',
               '/jury_box', '/kennel/indoor', '/kindergarden_classroom',
               '/kindergarden_classroom/n_kids_eating_or_drinking_or_talking',
               '/kindergarden_classroom/n_parents_or_instructors_playing_with_m_kids', '/kitchen',
               '/kitchen/n_people_cooking_food', '/kitchenette', '/lab_classroom', '/labyrinth/indoor', '/landing',
               '/laundromat', '/lavatory', '/lecture_room', '/legislative_chamber',
               '/legislative_chamber/a_representative_giving_a_speech', '/library/indoor', '/lido_deck/indoor',
               '/liquor_store/indoor', '/living_room', '/lobby', '/locker_room',
               '/locker_room/n_people_celebrating_a_victory', '/loft', '/lookout_station/indoor', '/machine_shop',
               '/martial_arts_gym', '/martial_arts_gym/n_people_stretching', '/maternity_ward', '/medina', '/mess_hall',
               '/mezzanine', '/mini_golf_course/indoor', '/moon_bounce', '/morgue', '/mosque/indoor',
               '/movie_theater/indoor', '/museum/indoor', '/museum_indoor/n_visitors_talking', '/music_store',
               '/music_studio', '/music_studio/n_musicians_listening_and_waiting_to_start_singing_or_playing',
               '/natural_history_museum', '/newsroom', '/nightclub', '/nursery', '/nursing_home', '/observatory/indoor',
               '/office', '/office/a_worker_making_a_pitch', '/office_cubicles', '/oil_refinery/indoor',
               '/operating_room', '/organ_loft/interior', '/orlop_deck', '/ossuary', '/outhouse/indoor', '/oyster_bar',
               '/palace_hall', '/pantry', '/parking_garage/indoor', '/parlor', '/particle_accelerator',
               '/party_tent/indoor', '/pawnshop', '/pedestrian_overpass/indoor', '/penalty_box', '/perfume_shop',
               '/pharmacy', '/physics_laboratory', '/piano_store', '/pig_farm', '/pilothouse/indoor', '/pizzeria',
               '/pizzeria/n_clients_ordering_pizza_from_the_table_or_counter', '/playroom', '/podium/indoor',
               '/podium/outdoor', '/poolroom/establishment', '/poolroom/home', '/portrait_studio', '/print_shop',
               '/promenade_deck', '/pub/indoor', '/pub_indoor/n_people_making_a_toast', '/pulpit', '/pump_room',
               '/quonset_hut/indoor', '/reading_room', '/reception', '/reception/n_receptionists_attending_m_people',
               '/recreation_room', '/recycling_plant/indoor', '/refectory', '/repair_shop', '/restaurant',
               '/restaurant/eating_and_drinking_and_talking', '/restaurant/ordering_food', '/restaurant/serving_food',
               '/restaurant_kitchen', '/restroom/indoor', '/riding_arena', '/roller_skating_rink/indoor',
               '/root_cellar', '/sacristy', '/sauna', '/science_museum', '/scriptorium', '/security_check_point',
               '/server_room', '/sewer', '/sewing_room', '/shipping_room', '/shoe_shop', '/shopping_mall/indoor',
               '/shower', '/shower_room', '/spa/massage_room', '/spa/mineral_bath', '/sporting_goods_store',
               '/squash_court', '/stable', '/stable/n_people_brushing_or_cleaning_or_preparing_a_horse',
               '/stable/n_people_taking_a_horse_out_of_the_stable_by_pulling_or_mounting_it', '/stage/indoor',
               '/staircase', '/starting_gate', '/storage_room', '/subway_interior', '/subway_station/corridor',
               '/subway_station/platform', '/supermarket', '/supermarket/n_people_handling_and_looking_at_a_product',
               '/sushi_bar', '/swimming_pool/indoor', '/tearoom', '/teashop', '/television_room',
               '/television_room/n_people_lying_on_the_couch', '/television_studio', '/tennis_court/indoor',
               '/theater/indoor_procenium', '/theater/indoor_round', '/theater/indoor_seats', '/thriftshop',
               '/throne_room', '/ticket_booth', '/ticket_window/indoor', '/tobacco_shop/indoor', '/topiary_garden',
               '/toyshop', '/trading_floor', '/turkish_bath', '/utility_room',
               '/utility_room/n_people_taking_clothes_out_of_the_laundry_machine', '/vestry', '/veterinarians_office',
               '/videostore', '/volleyball_court/indoor', '/volleyball_court_outdoor/a_player_blocking_a_shot',
               '/voting_booth', '/waiting_room', '/waiting_room/n_people_talking_to_other_people', '/walk_in_freezer',
               '/warehouse/indoor', '/washhouse/indoor', '/water_treatment_plant/indoor', '/wet_bar', '/window_seat',
               '/wine_cellar/barrel_storage', '/wine_cellar/bottle_storage', '/winery', '/witness_stand', '/workroom',
               '/workshop', '/wrestling_ring/indoor', '/youth_hostel', '/youth_hostel/n_people_setting_up_the_room',
               '/zen_garden', 'alcove', 'amusement_arcade', 'amusement_park', 'aquarium', 'arcade', 'arch', 'archive',
               'arena/rodeo', 'army_base', 'art_gallery', 'art_school', 'art_studio', 'artists_loft', 'atrium/public',
               'attic', 'auditorium', 'auto_showroom', 'ballroom', 'bank_vault', 'banquet_hall', 'bar', 'basement',
               'basketball_court/indoor', 'bathroom', 'beauty_salon', 'bedchamber', 'bedroom', 'beer_garden', 'berth',
               'biology_laboratory', 'bookstore', 'booth/indoor', 'bow_window/indoor', 'bowling_alley',
               'burial_chamber', 'bus_station/indoor', 'cafeteria', 'catacomb', 'chemistry_lab', 'childs_room',
               'church/indoor', 'classroom', 'clean_room', 'closet', 'coffee_shop', 'computer_room',
               'conference_center', 'conference_room', 'corridor', 'day_care_center', 'delicatessen', 'dining_hall',
               'dining_room', 'discotheque', 'doorway/outdoor', 'dorm_room', 'dressing_room', 'elevator/interior',
               'elevator_lobby', 'entrance_hall', 'escalator/indoor', 'excavation', 'fabric_store',
               'flea_market/indoor', 'florist_shop/indoor', 'food_court', 'garage/indoor', 'garage/outdoor',
               'gift_shop', 'gymnasium/indoor', 'hardware_store', 'home_office', 'home_theater', 'hospital',
               'hotel_room', 'ice_cream_parlor', 'isc', 'jacuzzi/indoor', 'jail_cell', 'jewelry_shop',
               'kindergarden_classroom', 'kiosk/indoor', 'kitchen', 'laundromat', 'lecture_room', 'legislative_chamber',
               'library/indoor', 'living_room', 'loading_dock', 'lobby', 'locker_room', 'martial_arts_gym', 'mezzanine',
               'movie_theater/indoor', 'museum/indoor', 'music_studio', 'natural_history_museum', 'nursery',
               'nursing_home', 'office', 'office_cubicles', 'operating_room', 'pantry', 'pet_shop', 'pharmacy',
               'playroom', 'poolroom/establishment', 'poolroom/home', 'porch', 'pub/indoor', 'reception',
               'recreation_room', 'repair_shop', 'restaurant', 'restaurant_kitchen', 'restaurant_patio', 'roof_garden',
               'science_museum', 'server_room', 'shower', 'staircase', 'storage_room', 'swimming_pool/indoor',
               'television_room', 'television_studio', 'theater/indoor_procenium', 'throne_room', 'toyshop',
               'utility_room', 'utliers/amphitheater_indoor', 'utliers/artists_loft/questionable',
               'utliers/assembly_hall', 'utliers/back_porch', 'utliers/backdrop', 'utliers/backroom',
               'utliers/backstairs_indoor', 'utliers/balustrade', 'utliers/bath_indoor', 'utliers/bookshelf',
               'utliers/breakfast_table', 'utliers/bubble_chamber', 'utliers/buffet', 'utliers/bulkhead',
               'utliers/bunk_bed', 'utliers/cabin_cruiser', 'utliers/cellar', 'utliers/cocktail_lounge',
               'utliers/deck-house_boat_deck_house', 'utliers/dining_area', 'utliers/embrasure',
               'utliers/entranceway_indoor', 'utliers/estaminet', 'utliers/flatlet', 'utliers/laboratorywet',
               'utliers/loge', 'utliers/loo', 'utliers/mental_institution/indoor', 'utliers/merlon', 'utliers/palestra',
               'utliers/powder_room', 'utliers/reception_room', 'utliers/retaining_wall', 'utliers/rotisserie',
               'utliers/salon', 'utliers/sanatorium', 'utliers/science_laboratory', 'utliers/scullery',
               'utliers/shelter_deck', 'utliers/snack_bar', 'utliers/stage_set', 'utliers/stall', 'utliers/stateroom',
               'utliers/store', 'utliers/student_center', 'utliers/study_hall', 'utliers/sunroom',
               'utliers/supply_chamber', 'utliers/tannery', 'utliers/teahouse', 'utliers/ticket_window/indoor',
               'utliers/tomb', 'utliers/upper_balcony', 'utliers/vat', 'utliers/vestibule', 'utliers/walkway',
               'utliers/war_room', 'utliers/washroom', 'utliers/widows_walk_indoor', 'veterinarians_office',
               'waiting_room', 'water_park', 'wet_bar', '~not labeled'}