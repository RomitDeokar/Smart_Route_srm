/* ════════════════════════════════════════════════════════════════
   _shared/cities.js — Comprehensive city data merged from GitHub
   src/index.tsx (RomitDeokar/Smart_Route_srm).

   Provides:
   • CITY_COORDS — lat/lon for all supported cities (incl. SRM campuses)
   • CITY_TOP_ATTRACTIONS — curated real top tourist places per city
   • CITY_DISTANCES — approx km between major Indian cities
   • CITY_IATA — airport codes
   • CAMPUS_MAP — university campus mapping
   • Helpers: haversineKm, lookupCoords, getDistance, getIATA, geocode, resolveCampus
   ════════════════════════════════════════════════════════════════ */

export const CITY_COORDS = {
  paris:[48.8566,2.3522],london:[51.5074,-0.1278],tokyo:[35.6762,139.6503],jaipur:[26.9124,75.7873],
  rome:[41.9028,12.4964],'new york':[40.7128,-74.006],dubai:[25.2048,55.2708],singapore:[1.3521,103.8198],
  bangkok:[13.7563,100.5018],barcelona:[41.3874,2.1686],istanbul:[41.0082,28.9784],amsterdam:[52.3676,4.9041],
  sydney:[-33.8688,151.2093],bali:[-8.3405,115.092],goa:[15.2993,74.124],udaipur:[24.5854,73.7125],
  varanasi:[25.3176,83.0068],mumbai:[19.076,72.8777],delhi:[28.7041,77.1025],agra:[27.1767,78.0081],
  chennai:[13.0827,80.2707],srm:[12.8231,80.0442],srmist:[12.8231,80.0442],kattankulathur:[12.8231,80.0442],
  mahabalipuram:[12.6169,80.1993],pondicherry:[11.9416,79.8083],bangalore:[12.9716,77.5946],
  hyderabad:[17.385,78.4867],kolkata:[22.5726,88.3639],lucknow:[26.8467,80.9462],kochi:[9.9312,76.2673],
  shimla:[31.1048,77.1734],manali:[32.2432,77.1892],ooty:[11.4102,76.6950],mysore:[12.2958,76.6394],
  coorg:[12.4244,75.7382],hampi:[15.335,76.46],munnar:[10.0889,77.0595],alleppey:[9.4981,76.3388],
  darjeeling:[27.041,88.2663],gangtok:[27.3389,88.6065],leh:[34.1526,77.5771],srinagar:[34.0837,74.7973],
  amritsar:[31.6340,74.8723],jodhpur:[26.2389,73.0243],pushkar:[26.4897,74.5511],ranthambore:[26.0173,76.5026],
  rishikesh:[30.0869,78.2676],haridwar:[29.9457,78.1642],tirupati:[13.6288,79.4192],rameshwaram:[9.2876,79.3129],
  madurai:[9.9252,78.1198],thanjavur:[10.787,79.1378],kodaikanal:[10.2381,77.4892],
  trichy:[10.7905,78.7047],tiruchirappalli:[10.7905,78.7047],
  'greater noida':[28.4744,77.5040],noida:[28.5355,77.3910],gurgaon:[28.4595,77.0266],
  amaravati:[16.5062,80.6480],vijayawada:[16.5062,80.6480],visakhapatnam:[17.6868,83.2185],
  chandigarh:[30.7333,76.7794],pune:[18.5204,73.8567],ahmedabad:[23.0225,72.5714],coimbatore:[11.0168,76.9558],
  thiruvananthapuram:[8.5241,76.9366],vellore:[12.9165,79.1325],
};

/* Curated top attractions per city — real, named tourist spots with coordinates,
   description, and Wikipedia titles. Used by the itinerary builder so major
   landmarks are always represented in plans. */
export const CITY_TOP_ATTRACTIONS = {
  chennai: [
    {name:'Marina Beach',lat:13.0500,lon:80.2824,type:'beach',description:'One of the longest urban beaches in the world, stretching 13 km along the Bay of Bengal.',wikiTitle:'Marina Beach'},
    {name:'Kapaleeshwarar Temple',lat:13.0339,lon:80.2694,type:'temple',description:'Ancient Dravidian-style Shiva temple dating back to the 7th century in Mylapore.',wikiTitle:'Kapaleeshwarar Temple'},
    {name:'Fort St. George',lat:13.0797,lon:80.2871,type:'fort',description:'First English fortress in India, built in 1644 by the East India Company.',wikiTitle:'Fort St. George, Chennai'},
    {name:'San Thome Cathedral',lat:13.0335,lon:80.2780,type:'historic',description:'A Roman Catholic cathedral built over the tomb of St. Thomas the Apostle.',wikiTitle:'San Thome Cathedral'},
    {name:'Government Museum Chennai',lat:13.0699,lon:80.2539,type:'museum',description:'Second oldest museum in India with a rich collection of archaeological artifacts.',wikiTitle:'Government Museum, Chennai'},
    {name:'Valluvar Kottam',lat:13.0508,lon:80.2345,type:'monument',description:'A monument dedicated to the Tamil poet Thiruvalluvar, shaped like a temple chariot.',wikiTitle:'Valluvar Kottam'},
    {name:'Elliot Beach',lat:13.0005,lon:80.2730,type:'beach',description:'A serene beach in Besant Nagar, popular with locals for evening walks.',wikiTitle:"Elliot's Beach"},
    {name:'DakshinaChitra Museum',lat:12.8168,lon:80.2261,type:'museum',description:'Living museum of art, architecture, and culture of South India.',wikiTitle:'DakshinaChitra'},
    {name:'Mahabalipuram Shore Temple',lat:12.6169,lon:80.1993,type:'temple',description:'UNESCO World Heritage Site — a 7th-century structural temple overlooking the Bay of Bengal.',wikiTitle:'Shore Temple'},
    {name:"Arjuna's Penance",lat:12.6165,lon:80.1946,type:'monument',description:"World's largest open-air bas-relief, a masterpiece of Pallava sculpture at Mahabalipuram.",wikiTitle:"Arjuna%27s Penance"},
    {name:'Guindy National Park',lat:13.0063,lon:80.2346,type:'park',description:'One of the few national parks inside a city, home to blackbuck and spotted deer.',wikiTitle:'Guindy National Park'},
    {name:'Madras Crocodile Bank',lat:12.7486,lon:80.2440,type:'park',description:'Famous reptile zoo on ECR with 2,500+ crocodiles, near Mahabalipuram.',wikiTitle:'Madras Crocodile Bank Trust'},
    {name:'VGP Universal Kingdom',lat:12.9395,lon:80.2502,type:'attraction',description:'Theme park on ECR with rides, water park and beach access.',wikiTitle:'VGP Universal Kingdom'},
    {name:'Five Rathas Mahabalipuram',lat:12.6133,lon:80.1933,type:'historic',description:'UNESCO Heritage rock-cut monolith temples carved during the Pallava reign.',wikiTitle:'Pancha Rathas'},
    {name:'Tiger Cave Mahabalipuram',lat:12.6390,lon:80.2025,type:'historic',description:'Ancient rock-cut shrine with a striking carving of 11 yali heads.',wikiTitle:'Tiger Cave'},
  ],
  // SRM-area POIs (within 25 km of SRMIST Kattankulathur 12.8231,80.0442) — used when destination resolves to Chennai SRM
  'chennai srm': [
    {name:'SRMIST Main Campus',lat:12.8231,lon:80.0442,type:'attraction',description:'Sprawling 250-acre SRM Institute of Science and Technology campus — the flagship SRM university.',wikiTitle:'SRM Institute of Science and Technology'},
    {name:'Mahabalipuram Shore Temple',lat:12.6169,lon:80.1993,type:'temple',description:'UNESCO Heritage 7th-century structural temple — 35 km drive on ECR.',wikiTitle:'Shore Temple'},
    {name:"Arjuna's Penance",lat:12.6165,lon:80.1946,type:'monument',description:"World's largest open-air bas-relief at Mahabalipuram — Pallava-era masterpiece.",wikiTitle:"Arjuna%27s Penance"},
    {name:'Five Rathas Mahabalipuram',lat:12.6133,lon:80.1933,type:'historic',description:'UNESCO Heritage rock-cut monolith temples carved during the Pallava reign.',wikiTitle:'Pancha Rathas'},
    {name:'Tiger Cave Mahabalipuram',lat:12.6390,lon:80.2025,type:'historic',description:'Ancient rock-cut shrine with a striking carving of 11 yali heads.',wikiTitle:'Tiger Cave'},
    {name:'DakshinaChitra Museum',lat:12.8168,lon:80.2261,type:'museum',description:'Living museum of art, architecture, and culture of South India — 15 km from SRM.',wikiTitle:'DakshinaChitra'},
    {name:'Madras Crocodile Bank',lat:12.7486,lon:80.2440,type:'park',description:'Reptile zoo with 2,500+ crocodiles on ECR — 15 km from SRM.',wikiTitle:'Madras Crocodile Bank Trust'},
    {name:'Muttukadu Boat House',lat:12.8126,lon:80.2447,type:'attraction',description:'Backwater boating, jet-ski, kayaking on the Muttukadu lagoon — 18 km from SRM.',wikiTitle:'Muttukadu'},
    {name:'Cholamandal Artists Village',lat:12.8779,lon:80.2469,type:'attraction',description:"India's largest self-supporting artists' colony, founded 1966 — galleries and beach.",wikiTitle:'Cholamandal Artists Village'},
    {name:'Vandalur Zoo (Arignar Anna)',lat:12.8923,lon:80.0825,type:'park',description:'1,500-acre zoological park with 1,200+ species — 8 km north of SRMIST.',wikiTitle:'Arignar Anna Zoological Park'},
    {name:'Thirukazhukundram Temple',lat:12.6045,lon:80.0651,type:'temple',description:'Hilltop Vedagiriswarar Temple, 25 km from SRM — famed for the eagle legend.',wikiTitle:'Tirukalukundram'},
    {name:'Kovalam Beach (ECR)',lat:12.7917,lon:80.2536,type:'beach',description:'Quiet ECR surf beach 18 km from SRM — popular with weekend day-trippers.',wikiTitle:'Kovalam, Chennai'},
    {name:'Mayajaal Multiplex',lat:12.8540,lon:80.2475,type:'attraction',description:'Entertainment complex with cinemas, gaming, bowling — 20 min from SRM.',wikiTitle:'Mayajaal'},
  ],
  jaipur: [
    {name:'Amber Fort',lat:26.9855,lon:75.8513,type:'fort',description:'Magnificent hilltop fort palace overlooking Maota Lake, built from red sandstone and marble.',wikiTitle:'Amber Fort'},
    {name:'Hawa Mahal',lat:26.9239,lon:75.8267,type:'palace',description:'Iconic Palace of Winds with 953 small windows designed for royal women to observe street life.',wikiTitle:'Hawa Mahal'},
    {name:'City Palace Jaipur',lat:26.9258,lon:75.8237,type:'palace',description:'Grand palace complex blending Mughal and Rajput architecture, still home to the royal family.',wikiTitle:'City Palace, Jaipur'},
    {name:'Jantar Mantar',lat:26.9247,lon:75.8241,type:'monument',description:"UNESCO World Heritage astronomical observation site with the world's largest sundial.",wikiTitle:'Jantar Mantar, Jaipur'},
    {name:'Nahargarh Fort',lat:26.9378,lon:75.8150,type:'fort',description:'Hilltop fort offering stunning panoramic views of the Pink City, especially at sunset.',wikiTitle:'Nahargarh Fort'},
    {name:'Jaigarh Fort',lat:26.9864,lon:75.8427,type:'fort',description:"Fort housing Jaivana, the world's largest cannon on wheels.",wikiTitle:'Jaigarh Fort'},
    {name:'Albert Hall Museum',lat:26.9117,lon:75.8190,type:'museum',description:'Indo-Saracenic architecture museum housing Egyptian mummy and ancient artifacts.',wikiTitle:'Albert Hall Museum'},
    {name:'Jal Mahal',lat:26.9530,lon:75.8466,type:'palace',description:'Ethereal floating palace in the middle of Man Sagar Lake.',wikiTitle:'Jal Mahal'},
    {name:'Birla Mandir Jaipur',lat:26.8923,lon:75.8150,type:'temple',description:'Beautiful white marble temple dedicated to Lord Vishnu and Goddess Lakshmi.',wikiTitle:'Birla Mandir, Jaipur'},
    {name:'Johari Bazaar',lat:26.9213,lon:75.8269,type:'market',description:'Famous jewelry and textile market in the heart of the Pink City.',wikiTitle:'Johari Bazaar'},
  ],
  goa: [
    {name:'Calangute Beach',lat:15.5441,lon:73.7554,type:'beach',description:'The largest beach in North Goa, known as the Queen of Beaches.',wikiTitle:'Calangute'},
    {name:'Fort Aguada',lat:15.4920,lon:73.7738,type:'fort',description:'17th-century Portuguese fort with a lighthouse overlooking the Arabian Sea.',wikiTitle:'Fort Aguada'},
    {name:'Basilica of Bom Jesus',lat:15.5009,lon:73.9116,type:'historic',description:'UNESCO World Heritage Site housing the remains of St. Francis Xavier.',wikiTitle:'Basilica of Bom Jesus'},
    {name:'Dudhsagar Falls',lat:15.3144,lon:74.3143,type:'viewpoint',description:"Four-tiered waterfall on the Mandovi River, one of India's tallest at 310m.",wikiTitle:'Dudhsagar Falls'},
    {name:'Anjuna Beach',lat:15.5741,lon:73.7412,type:'beach',description:'Famous for its Wednesday flea market and vibrant nightlife.',wikiTitle:'Anjuna'},
    {name:'Se Cathedral',lat:15.5039,lon:73.9128,type:'historic',description:'One of the largest churches in Asia, built in Portuguese-Gothic style.',wikiTitle:'Se Cathedral of Goa'},
    {name:'Palolem Beach',lat:15.0099,lon:74.0235,type:'beach',description:'Crescent-shaped beach in South Goa known for its calm waters and beauty.',wikiTitle:'Palolem'},
    {name:'Baga Beach',lat:15.5563,lon:73.7513,type:'beach',description:'Popular beach famous for water sports, nightlife, and shack culture.',wikiTitle:'Baga Beach'},
  ],
  delhi: [
    {name:'Red Fort',lat:28.6562,lon:77.2410,type:'fort',description:"UNESCO World Heritage Mughal fort, India's Independence Day celebrations venue.",wikiTitle:'Red Fort'},
    {name:'Qutub Minar',lat:28.5245,lon:77.1855,type:'monument',description:'UNESCO site — tallest brick minaret in the world at 72.5 meters.',wikiTitle:'Qutub Minar'},
    {name:'India Gate',lat:28.6129,lon:77.2295,type:'monument',description:'Iconic 42m war memorial arch on Rajpath, central landmark of Delhi.',wikiTitle:'India Gate'},
    {name:"Humayun's Tomb",lat:28.5933,lon:77.2507,type:'monument',description:'UNESCO Heritage — inspiration for the Taj Mahal, set in beautiful gardens.',wikiTitle:"Humayun%27s Tomb"},
    {name:'Lotus Temple',lat:28.5535,lon:77.2588,type:'temple',description:"Baha'i House of Worship shaped like a lotus flower, architectural marvel.",wikiTitle:'Lotus Temple'},
    {name:'Jama Masjid',lat:28.6507,lon:77.2334,type:'historic',description:"India's largest mosque, built by Shah Jahan with stunning red sandstone.",wikiTitle:'Jama Masjid, Delhi'},
    {name:'Akshardham Temple',lat:28.6127,lon:77.2773,type:'temple',description:'Spectacular Hindu temple complex showcasing 10,000 years of Indian culture.',wikiTitle:'Akshardham (Delhi)'},
    {name:'Chandni Chowk',lat:28.6506,lon:77.2302,type:'market',description:"One of India's oldest and busiest markets, famous for street food.",wikiTitle:'Chandni Chowk'},
    {name:'Lodhi Garden',lat:28.5935,lon:77.2197,type:'park',description:'Historic park with 15th-century Mughal tombs spread over 90 acres.',wikiTitle:'Lodhi Garden'},
  ],
  mumbai: [
    {name:'Gateway of India',lat:18.9220,lon:72.8347,type:'monument',description:"Iconic arch monument built in 1924 to commemorate King George V's visit.",wikiTitle:'Gateway of India'},
    {name:'Marine Drive',lat:18.9432,lon:72.8235,type:'viewpoint',description:"3.6 km promenade along the coast, known as the Queen's Necklace at night.",wikiTitle:'Marine Drive, Mumbai'},
    {name:'Elephanta Caves',lat:18.9633,lon:72.9315,type:'historic',description:'UNESCO Heritage cave temples dedicated to Lord Shiva on Elephanta Island.',wikiTitle:'Elephanta Caves'},
    {name:'Chhatrapati Shivaji Terminus',lat:18.9398,lon:72.8355,type:'historic',description:'UNESCO World Heritage Victorian Gothic railway station.',wikiTitle:'Chhatrapati Shivaji Maharaj Terminus'},
    {name:'Juhu Beach',lat:19.0989,lon:72.8269,type:'beach',description:'Famous beach known for street food, sunset views, and Bollywood spotting.',wikiTitle:'Juhu Beach'},
    {name:'Haji Ali Dargah',lat:18.9827,lon:72.8089,type:'temple',description:'Iconic mosque built on an islet, accessible only during low tide.',wikiTitle:'Haji Ali Dargah'},
    {name:'Siddhivinayak Temple',lat:19.0166,lon:72.8300,type:'temple',description:'One of the richest and most visited Ganesh temples in Mumbai.',wikiTitle:'Siddhivinayak Temple'},
    {name:'Crawford Market',lat:18.9475,lon:72.8344,type:'market',description:'Historic market with Norman Gothic architecture, bustling with local culture.',wikiTitle:'Mahatma Jyotiba Phule Mandai'},
  ],
  agra: [
    {name:'Taj Mahal',lat:27.1751,lon:78.0421,type:'monument',description:'UNESCO World Heritage — an ivory-white marble mausoleum, one of the Seven Wonders.',wikiTitle:'Taj Mahal'},
    {name:'Agra Fort',lat:27.1795,lon:78.0211,type:'fort',description:'UNESCO Heritage red sandstone fort with white marble palaces inside.',wikiTitle:'Agra Fort'},
    {name:'Fatehpur Sikri',lat:27.0945,lon:77.6679,type:'historic',description:'UNESCO Heritage — abandoned Mughal city built by Emperor Akbar.',wikiTitle:'Fatehpur Sikri'},
    {name:'Itimad-ud-Daulah',lat:27.1925,lon:78.0312,type:'monument',description:'Known as Baby Taj, an exquisite white marble Mughal tomb.',wikiTitle:"Tomb of I%27timad-ud-Daulah"},
    {name:'Mehtab Bagh',lat:27.1800,lon:78.0444,type:'park',description:'Mughal garden with stunning views of the Taj Mahal across the Yamuna.',wikiTitle:'Mehtab Bagh'},
  ],
  varanasi: [
    {name:'Dashashwamedh Ghat',lat:25.3048,lon:83.0108,type:'historic',description:'The main ghat famous for its spectacular evening Ganga Aarti ceremony.',wikiTitle:'Dashashwamedh Ghat'},
    {name:'Kashi Vishwanath Temple',lat:25.3109,lon:83.0107,type:'temple',description:'One of the most revered Hindu temples dedicated to Lord Shiva.',wikiTitle:'Kashi Vishwanath Temple'},
    {name:'Sarnath',lat:25.3814,lon:83.0224,type:'historic',description:'Buddhist pilgrimage site where Buddha gave his first sermon.',wikiTitle:'Sarnath'},
    {name:'Assi Ghat',lat:25.2856,lon:83.0063,type:'historic',description:'The southernmost ghat of Varanasi, important pilgrimage and cultural spot.',wikiTitle:'Assi Ghat'},
    {name:'Manikarnika Ghat',lat:25.3128,lon:83.0120,type:'historic',description:'The primary cremation ghat, considered the most sacred in Hinduism.',wikiTitle:'Manikarnika Ghat'},
    {name:'Ramnagar Fort',lat:25.2866,lon:83.0289,type:'fort',description:'18th-century fort and palace of the Maharaja of Varanasi.',wikiTitle:'Ramnagar Fort'},
  ],
  kolkata: [
    {name:'Victoria Memorial',lat:22.5448,lon:88.3426,type:'monument',description:'Magnificent white marble hall and museum dedicated to Queen Victoria.',wikiTitle:'Victoria Memorial, Kolkata'},
    {name:'Howrah Bridge',lat:22.5851,lon:88.3468,type:'monument',description:'Iconic cantilever bridge over the Hooghly River, a symbol of Kolkata.',wikiTitle:'Howrah Bridge'},
    {name:'Indian Museum',lat:22.5583,lon:88.3508,type:'museum',description:'The oldest and largest museum in India with rare collections.',wikiTitle:'Indian Museum'},
    {name:'Dakshineswar Kali Temple',lat:22.6551,lon:88.3577,type:'temple',description:'Famous temple associated with Ramakrishna Paramahamsa.',wikiTitle:'Dakshineswar Kali Temple'},
    {name:'Park Street',lat:22.5520,lon:88.3599,type:'market',description:'Historic boulevard known for restaurants, nightlife and colonial architecture.',wikiTitle:'Park Street, Kolkata'},
  ],
  udaipur: [
    {name:'City Palace Udaipur',lat:24.5764,lon:73.6915,type:'palace',description:'Sprawling palace complex on the banks of Lake Pichola, a must-visit.',wikiTitle:'City Palace, Udaipur'},
    {name:'Lake Pichola',lat:24.5720,lon:73.6809,type:'viewpoint',description:'Beautiful artificial lake with Lake Palace Hotel seemingly floating on it.',wikiTitle:'Lake Pichola'},
    {name:'Jag Mandir',lat:24.5686,lon:73.6876,type:'palace',description:'Island palace on Lake Pichola, used as a summer resort by royals.',wikiTitle:'Jag Mandir'},
    {name:'Sajjangarh Palace',lat:24.5770,lon:73.6485,type:'palace',description:'Hilltop Monsoon Palace with panoramic views of the City of Lakes.',wikiTitle:'Monsoon Palace'},
    {name:'Saheliyon ki Bari',lat:24.5912,lon:73.7022,type:'garden',description:'Garden of the Maidens with fountains, kiosks, marble elephants.',wikiTitle:'Saheliyon-ki-Bari'},
  ],
  bangalore: [
    {name:'Lalbagh Botanical Garden',lat:12.9507,lon:77.5848,type:'park',description:'Sprawling botanical garden with a famous glass house and centuries-old trees.',wikiTitle:'Lal Bagh'},
    {name:'Bangalore Palace',lat:12.9987,lon:77.5922,type:'palace',description:'Tudor-style palace inspired by Windsor Castle with fortified towers.',wikiTitle:'Bangalore Palace'},
    {name:'Cubbon Park',lat:12.9763,lon:77.5929,type:'park',description:'120-year-old park in the heart of Bangalore with 6000+ trees.',wikiTitle:'Cubbon Park'},
    {name:'ISKCON Temple Bangalore',lat:12.9715,lon:77.5511,type:'temple',description:'One of the largest ISKCON temples in the world.',wikiTitle:'ISKCON Temple Bangalore'},
    {name:'Tipu Sultan Palace',lat:12.9592,lon:77.5737,type:'palace',description:'Summer palace of Tipu Sultan built in Indo-Islamic style.',wikiTitle:"Tipu Sultan%27s Summer Palace"},
    {name:'Nandi Hills',lat:13.3702,lon:77.6835,type:'viewpoint',description:'Hill station 60km from Bangalore, famous for sunrise and paragliding.',wikiTitle:'Nandi Hills'},
  ],
  hyderabad: [
    {name:'Charminar',lat:17.3616,lon:78.4747,type:'monument',description:'Iconic 16th-century monument and mosque, symbol of Hyderabad.',wikiTitle:'Charminar'},
    {name:'Golconda Fort',lat:17.3833,lon:78.4011,type:'fort',description:'Massive medieval fort known for its acoustic architecture.',wikiTitle:'Golconda'},
    {name:'Ramoji Film City',lat:17.2543,lon:78.6808,type:'attraction',description:"World's largest integrated film studio complex and theme park.",wikiTitle:'Ramoji Film City'},
    {name:'Hussain Sagar Lake',lat:17.4239,lon:78.4738,type:'viewpoint',description:'Heart-shaped lake with a monolithic Buddha statue in the center.',wikiTitle:'Hussain Sagar'},
    {name:'Salar Jung Museum',lat:17.3714,lon:78.4804,type:'museum',description:'One of the largest one-man collections of art in the world.',wikiTitle:'Salar Jung Museum'},
  ],
  pondicherry: [
    {name:'Promenade Beach',lat:11.9327,lon:79.8369,type:'beach',description:'1.5 km rocky beach along the Bay of Bengal in the French Quarter.',wikiTitle:'Promenade Beach'},
    {name:'Auroville',lat:12.0063,lon:79.8108,type:'attraction',description:'Experimental universal township with the iconic golden Matrimandir.',wikiTitle:'Auroville'},
    {name:'French Quarter',lat:11.9340,lon:79.8370,type:'historic',description:'Charming colonial area with French architecture, cafes, and boutiques.',wikiTitle:'White Town, Pondicherry'},
    {name:'Paradise Beach',lat:11.9008,lon:79.8369,type:'beach',description:'Secluded golden sand beach accessible only by boat.',wikiTitle:'Paradise Beach, Pondicherry'},
    {name:'Sri Aurobindo Ashram',lat:11.9353,lon:79.8365,type:'temple',description:'Spiritual community founded by Sri Aurobindo and The Mother.',wikiTitle:'Sri Aurobindo Ashram'},
  ],
  kochi: [
    {name:'Fort Kochi',lat:9.9638,lon:76.2432,type:'historic',description:'Historic area with colonial architecture, churches, and Chinese fishing nets.',wikiTitle:'Fort Kochi'},
    {name:'Chinese Fishing Nets',lat:9.9676,lon:76.2279,type:'attraction',description:'Iconic cantilevered fishing nets introduced by Chinese explorers.',wikiTitle:'Chinese fishing nets'},
    {name:'Mattancherry Palace',lat:9.9582,lon:76.2597,type:'palace',description:'Dutch Palace with stunning Kerala murals depicting Hindu temple art.',wikiTitle:'Mattancherry Palace'},
    {name:'St. Francis Church',lat:9.9641,lon:76.2418,type:'historic',description:'Oldest European church in India, originally built in 1503.',wikiTitle:"St. Francis Church, Kochi"},
    {name:'Jew Town Kochi',lat:9.9572,lon:76.2602,type:'market',description:'Historic area with a 16th-century synagogue and antique shops.',wikiTitle:'Paradesi Synagogue'},
  ],
  trichy: [
    {name:'Rockfort Temple',lat:10.8085,lon:78.6946,type:'temple',description:'Ancient rock-cut temple atop a 83m rock, iconic landmark of Tiruchirappalli.',wikiTitle:'Rockfort'},
    {name:'Sri Ranganathaswamy Temple',lat:10.8627,lon:78.6892,type:'temple',description:'One of the largest functioning Hindu temples in the world, dedicated to Lord Vishnu.',wikiTitle:'Ranganathaswamy Temple, Srirangam'},
    {name:'Jambukeswarar Temple',lat:10.8537,lon:78.7072,type:'temple',description:'Ancient Shiva temple on Srirangam island, one of the Pancha Bhootha Sthalams.',wikiTitle:'Jambukeswarar Temple, Thiruvanaikaval'},
    {name:'Ucchi Pillayar Temple',lat:10.8090,lon:78.6950,type:'temple',description:'Temple dedicated to Lord Ganesha at the top of Rock Fort with panoramic views.',wikiTitle:'Ucchi Pillayar Temple'},
    {name:'Kallanai Dam',lat:10.8319,lon:78.8289,type:'historic',description:'Grand Anicut — one of the oldest water-diversion structures in the world, built by Cholas.',wikiTitle:'Kallanai'},
    {name:'Government Museum Trichy',lat:10.8052,lon:78.6887,type:'museum',description:'Museum housing ancient artifacts, sculptures, and geological specimens.',wikiTitle:'Government Museum, Tiruchirappalli'},
  ],
  'greater noida': [
    {name:'India Expo Centre',lat:28.4611,lon:77.5133,type:'attraction',description:'One of the largest exhibition centers in South Asia.',wikiTitle:'India Expo Centre and Mart'},
    {name:'Buddh International Circuit',lat:28.3484,lon:77.5338,type:'attraction',description:'Formula 1 racing circuit, one of the finest in Asia.',wikiTitle:'Buddh International Circuit'},
    {name:'Surajpur Bird Sanctuary',lat:28.5017,lon:77.5033,type:'park',description:'Wetland bird sanctuary with over 180 bird species.',wikiTitle:'Surajpur Wetland'},
    {name:'Great India Place Mall',lat:28.5686,lon:77.3234,type:'market',description:'One of the largest malls in North India with entertainment and shopping.',wikiTitle:'The Great India Place'},
    {name:'Akshardham Temple',lat:28.6127,lon:77.2773,type:'temple',description:'Spectacular Hindu temple complex showcasing Indian culture (nearby in Delhi).',wikiTitle:'Akshardham (Delhi)'},
    {name:'Worlds of Wonder',lat:28.5686,lon:77.3234,type:'attraction',description:'Amusement and water park with thrilling rides.',wikiTitle:'Worlds of Wonder (amusement park)'},
  ],
  amaravati: [
    {name:'Amaravati Stupa',lat:16.5725,lon:80.3572,type:'monument',description:'Ancient Buddhist stupa, one of the most important Buddhist sites in India.',wikiTitle:'Amaravati Stupa'},
    {name:'Undavalli Caves',lat:16.4961,lon:80.5810,type:'historic',description:'Rock-cut cave temples dating to 4th-5th century with monolithic Vishnu statue.',wikiTitle:'Undavalli Caves'},
    {name:'Prakasam Barrage',lat:16.5086,lon:80.6148,type:'viewpoint',description:'Dam across Krishna River connecting Vijayawada and Guntur.',wikiTitle:'Prakasam Barrage'},
    {name:'Kanaka Durga Temple',lat:16.5170,lon:80.6095,type:'temple',description:'Famous hilltop temple dedicated to Goddess Durga on Indrakeeladri hill.',wikiTitle:'Kanaka Durga Temple'},
    {name:'Bhavani Island',lat:16.5106,lon:80.5972,type:'attraction',description:'Largest river island in Krishna river with boating and water sports.',wikiTitle:'Bhavani Island'},
    {name:'Mangalagiri Temple',lat:16.4319,lon:80.5619,type:'temple',description:'Ancient hilltop temple dedicated to Lord Narasimha.',wikiTitle:'Mangalagiri'},
  ],
  manali: [
    {name:'Hadimba Temple',lat:32.2484,lon:77.1855,type:'temple',description:'Ancient cave temple dedicated to Hidimba Devi, set amid towering deodar forests.',wikiTitle:'Hidimba Devi Temple'},
    {name:'Solang Valley',lat:32.3169,lon:77.1567,type:'viewpoint',description:'Picturesque snow-point valley famous for paragliding, skiing, and zorbing.',wikiTitle:'Solang Valley'},
    {name:'Rohtang Pass',lat:32.3725,lon:77.2467,type:'viewpoint',description:'High mountain pass at 3,978m offering dramatic Himalayan views and snow year-round.',wikiTitle:'Rohtang Pass'},
    {name:'Old Manali',lat:32.2530,lon:77.1810,type:'historic',description:'Charming village area with cafes, boutique shops and apple orchards.',wikiTitle:'Manali'},
    {name:'Manu Temple',lat:32.2563,lon:77.1822,type:'temple',description:'Ancient temple dedicated to sage Manu.',wikiTitle:'Manu Temple'},
    {name:'Vashisht Hot Springs',lat:32.2691,lon:77.1881,type:'attraction',description:'Natural hot sulphur springs in a 4000-year-old village near Manali.',wikiTitle:'Vashisht'},
  ],
  shimla: [
    {name:'The Ridge',lat:31.1048,lon:77.1734,type:'viewpoint',description:'Large open street running east-west along the top of Shimla.',wikiTitle:'The Ridge, Shimla'},
    {name:'Mall Road',lat:31.1033,lon:77.1722,type:'market',description:'Famous shopping street with British-era buildings.',wikiTitle:'Mall Road, Shimla'},
    {name:'Jakhoo Temple',lat:31.1019,lon:77.1853,type:'temple',description:"Ancient Hanuman temple at Shimla's highest peak with a 108-foot tall statue.",wikiTitle:'Jakhu Temple'},
    {name:'Christ Church Shimla',lat:31.1037,lon:77.1729,type:'historic',description:'Second oldest church in North India.',wikiTitle:'Christ Church, Shimla'},
    {name:'Kufri',lat:31.0980,lon:77.2640,type:'viewpoint',description:'Hill station 16km from Shimla, famous for skiing and adventure sports.',wikiTitle:'Kufri'},
  ],
  darjeeling: [
    {name:'Tiger Hill',lat:27.0008,lon:88.2747,type:'viewpoint',description:'Famous sunrise viewpoint over Mt. Kanchenjunga.',wikiTitle:'Tiger Hill, Darjeeling'},
    {name:'Darjeeling Himalayan Railway',lat:27.0410,lon:88.2663,type:'historic',description:'UNESCO World Heritage toy train running narrow-gauge.',wikiTitle:'Darjeeling Himalayan Railway'},
    {name:'Padmaja Naidu Himalayan Zoological Park',lat:27.0496,lon:88.2611,type:'park',description:'Specialized zoo for Himalayan species.',wikiTitle:'Padmaja Naidu Himalayan Zoological Park'},
    {name:'Batasia Loop',lat:27.0287,lon:88.2622,type:'viewpoint',description:'Spiral railway loop with a war memorial.',wikiTitle:'Batasia Loop'},
    {name:'Happy Valley Tea Estate',lat:27.0505,lon:88.2545,type:'attraction',description:'One of the oldest tea estates in Darjeeling.',wikiTitle:'Happy Valley Tea Estate'},
  ],
  rishikesh: [
    {name:'Laxman Jhula',lat:30.1280,lon:78.3257,type:'historic',description:'Iconic suspension bridge across the Ganges.',wikiTitle:'Lakshman Jhula'},
    {name:'Ram Jhula',lat:30.1208,lon:78.3203,type:'historic',description:'Suspension bridge connecting two ashram-laden banks.',wikiTitle:'Ram Jhula'},
    {name:'Triveni Ghat',lat:30.1086,lon:78.3105,type:'historic',description:'Sacred bathing ghat where the famous evening Ganga Aarti is held daily.',wikiTitle:'Triveni Ghat'},
    {name:'The Beatles Ashram',lat:30.1153,lon:78.3225,type:'attraction',description:'Abandoned Maharishi Mahesh Yogi ashram.',wikiTitle:'Chaurasi Kutia'},
    {name:'Neelkanth Mahadev Temple',lat:30.1467,lon:78.3997,type:'temple',description:'Sacred Shiva temple set among forested hills.',wikiTitle:'Neelkanth Mahadev Temple'},
    {name:'Parmarth Niketan',lat:30.1186,lon:78.3231,type:'temple',description:'Largest yoga ashram in Rishikesh on the banks of the Ganges.',wikiTitle:'Parmarth Niketan'},
  ],
  ooty: [
    {name:'Ooty Lake',lat:11.4023,lon:76.6932,type:'viewpoint',description:'Artificial lake built in 1824, famous for boating amid eucalyptus trees.',wikiTitle:'Ooty Lake'},
    {name:'Botanical Gardens',lat:11.4133,lon:76.7050,type:'park',description:'55-acre gardens with rare plants.',wikiTitle:'Government Botanical Garden, Udagamandalam'},
    {name:'Doddabetta Peak',lat:11.4017,lon:76.7411,type:'viewpoint',description:'Highest peak in the Nilgiris at 2,637m.',wikiTitle:'Doddabetta'},
    {name:'Nilgiri Mountain Railway',lat:11.4102,lon:76.6950,type:'historic',description:'UNESCO World Heritage steam railway.',wikiTitle:'Nilgiri Mountain Railway'},
  ],
  pune: [
    {name:'Shaniwar Wada',lat:18.5196,lon:73.8553,type:'historic',description:'Historic 18th-century fortification, seat of the Peshwas of the Maratha Empire.',wikiTitle:'Shaniwar Wada'},
    {name:'Aga Khan Palace',lat:18.5526,lon:73.9006,type:'palace',description:'Historic palace where Mahatma Gandhi was interned.',wikiTitle:'Aga Khan Palace'},
    {name:'Sinhagad Fort',lat:18.3664,lon:73.7553,type:'fort',description:'Hilltop fortress 30km southwest of Pune.',wikiTitle:'Sinhagad'},
    {name:'Dagadusheth Halwai Ganpati Temple',lat:18.5161,lon:73.8567,type:'temple',description:'Famous Ganesh temple visited by celebrities and politicians.',wikiTitle:'Dagadusheth Halwai Ganapati'},
  ],
  amritsar: [
    {name:'Golden Temple',lat:31.6200,lon:74.8765,type:'temple',description:'Holiest gurdwara of Sikhism, gilded with real gold.',wikiTitle:'Golden Temple'},
    {name:'Wagah Border',lat:31.6044,lon:74.5723,type:'historic',description:'Daily Beating Retreat ceremony at India-Pakistan border.',wikiTitle:'Wagah'},
    {name:'Jallianwala Bagh',lat:31.6207,lon:74.8800,type:'historic',description:'Memorial site of the 1919 massacre.',wikiTitle:'Jallianwala Bagh'},
    {name:'Partition Museum',lat:31.6240,lon:74.8765,type:'museum',description:'World\u2019s first museum dedicated to the 1947 Partition of India.',wikiTitle:'Partition Museum'},
  ],
  alleppey: [
    {name:'Alappuzha Beach',lat:9.4907,lon:76.3293,type:'beach',description:'Famous beach with a 137-year-old pier.',wikiTitle:'Alappuzha Beach'},
    {name:'Backwater Houseboat Cruise',lat:9.4981,lon:76.3388,type:'attraction',description:'Iconic Kerala houseboat experience through palm-lined canals.',wikiTitle:'Kerala backwaters'},
    {name:'Krishnapuram Palace',lat:9.1731,lon:76.4944,type:'palace',description:'18th-century palace with the largest single-band mural in Kerala.',wikiTitle:'Krishnapuram Palace'},
    {name:'Marari Beach',lat:9.6076,lon:76.3047,type:'beach',description:'Quiet, palm-fringed beach 11km from Alleppey.',wikiTitle:'Mararikulam'},
  ],
  munnar: [
    {name:'Eravikulam National Park',lat:10.1809,lon:77.0617,type:'park',description:'Home of the Nilgiri Tahr.',wikiTitle:'Eravikulam National Park'},
    {name:'Tea Plantations',lat:10.0889,lon:77.0595,type:'attraction',description:'Endless rolling tea gardens — Munnar\u2019s defining landscape.',wikiTitle:'Munnar'},
    {name:'Mattupetty Dam',lat:10.1107,lon:77.1247,type:'viewpoint',description:'Concrete gravity dam with boating on the reservoir.',wikiTitle:'Mattupetty Dam'},
    {name:'Anamudi Peak',lat:10.1700,lon:77.0667,type:'viewpoint',description:'Highest peak in South India at 2,695m.',wikiTitle:'Anamudi'},
  ],
  hampi: [
    {name:'Virupaksha Temple',lat:15.3354,lon:76.4583,type:'temple',description:'Active 7th-century temple dedicated to Lord Shiva.',wikiTitle:'Virupaksha Temple, Hampi'},
    {name:'Vittala Temple',lat:15.3424,lon:76.4754,type:'temple',description:'Iconic 16th-century temple with the famous stone chariot.',wikiTitle:'Vittala Temple, Hampi'},
    {name:'Royal Enclosure',lat:15.3253,lon:76.4624,type:'historic',description:'Ruins of the royal palace complex of the Vijayanagara emperors.',wikiTitle:'Hampi'},
    {name:'Hemakuta Hill',lat:15.3339,lon:76.4595,type:'viewpoint',description:'Hill with cluster of pre-Vijayanagara temples and panoramic views.',wikiTitle:'Hampi'},
  ],
  leh: [
    {name:'Pangong Lake',lat:33.7544,lon:78.6450,type:'viewpoint',description:'Stunning high-altitude lake stretching from India to China.',wikiTitle:'Pangong Tso'},
    {name:'Nubra Valley',lat:34.5806,lon:77.5722,type:'viewpoint',description:'High-altitude desert valley with sand dunes and double-humped camels.',wikiTitle:'Nubra'},
    {name:'Khardung La',lat:34.2785,lon:77.6044,type:'viewpoint',description:'One of the highest motorable mountain passes in the world.',wikiTitle:'Khardung La'},
    {name:'Thiksey Monastery',lat:34.0589,lon:77.6688,type:'temple',description:'12-storey monastery resembling Lhasa\u2019s Potala Palace.',wikiTitle:'Thikse Monastery'},
    {name:'Hemis Monastery',lat:33.9133,lon:77.7050,type:'temple',description:'Largest Buddhist monastery in Ladakh.',wikiTitle:'Hemis Monastery'},
  ],
};

/* Approximate distances between major Indian cities (km) for price estimation. */
export const CITY_DISTANCES = {
  chennai: {delhi:2180,mumbai:1340,jaipur:2000,goa:870,bangalore:350,hyderabad:630,kolkata:1660,agra:2100,varanasi:1680,udaipur:1670,kochi:690,shimla:2550,manali:2700,pondicherry:150,amritsar:2600,jodhpur:1950,leh:3200,darjeeling:1900,ooty:540,mysore:480,mahabalipuram:60,madurai:460,thanjavur:340,kodaikanal:530,rishikesh:2350,hampi:580,munnar:600,alleppey:760,tirupati:140,srinagar:3100,coimbatore:500},
  delhi: {mumbai:1400,jaipur:280,goa:1900,bangalore:2150,hyderabad:1500,kolkata:1500,agra:230,varanasi:820,udaipur:670,kochi:2700,shimla:350,manali:530,chennai:2180,pondicherry:2300,amritsar:470,jodhpur:590,leh:1000,darjeeling:1550,rishikesh:250,haridwar:220,srinagar:850,coimbatore:2440},
  mumbai: {jaipur:1150,goa:590,bangalore:980,hyderabad:710,kolkata:2050,agra:1220,varanasi:1330,udaipur:660,kochi:1500,delhi:1400,chennai:1340,pondicherry:1490,amritsar:1840,jodhpur:830,shimla:1750,manali:1850,coimbatore:1320},
  bangalore: {mysore:150,ooty:275,kochi:550,chennai:350,hyderabad:570,goa:560,mumbai:980,hampi:340,coorg:250,coimbatore:370,madurai:430,delhi:2150},
  kolkata: {darjeeling:600,gangtok:640,varanasi:680,delhi:1500,chennai:1660,mumbai:2050},
  hyderabad: {chennai:630,bangalore:570,goa:660,mumbai:710,delhi:1500,kolkata:1500,vijayawada:275,vizag:610},
  jaipur: {delhi:280,agra:240,udaipur:400,jodhpur:340,mumbai:1150,bangalore:1900,chennai:2000,goa:1750},
  goa: {mumbai:590,bangalore:560,chennai:870,hyderabad:660,delhi:1900,jaipur:1750},
};

/* IATA codes for major Indian cities. */
export const CITY_IATA = {
  delhi:'DEL', mumbai:'BOM', bangalore:'BLR', bengaluru:'BLR', chennai:'MAA', kolkata:'CCU',
  hyderabad:'HYD', ahmedabad:'AMD', pune:'PNQ', goa:'GOI', kochi:'COK', cochin:'COK',
  jaipur:'JAI', lucknow:'LKO', trivandrum:'TRV', thiruvananthapuram:'TRV', coimbatore:'CJB',
  guwahati:'GAU', bhubaneswar:'BBI', indore:'IDR', nagpur:'NAG', patna:'PAT', srinagar:'SXR',
  amritsar:'ATQ', varanasi:'VNS', vishakhapatnam:'VTZ', visakhapatnam:'VTZ', vizag:'VTZ',
  agra:'AGR', udaipur:'UDR', jodhpur:'JDH', leh:'IXL', mangalore:'IXE', madurai:'IXM',
  tiruchirappalli:'TRZ', trichy:'TRZ', dehradun:'DED', chandigarh:'IXC',
  ranchi:'IXR', raipur:'RPR', bhopal:'BHO', jammu:'IXJ', surat:'STV', vadodara:'BDQ',
  pondicherry:'PNY', kannur:'CNN', tirupati:'TIR', rajkot:'RAJ', aurangabad:'IXU',
};

/* University / institution campus map. */
export const CAMPUS_MAP = {
  'srm university':{city:'Chennai',lat:12.8231,lon:80.0442,label:'SRM University, Kattankulathur (Chennai)'},
  'srm kattankulathur':{city:'Chennai',lat:12.8231,lon:80.0442,label:'SRM Kattankulathur Campus (Chennai)'},
  'srmist':{city:'Chennai',lat:12.8231,lon:80.0442,label:'SRMIST Main Campus (Chennai)'},
  'srm chennai':{city:'Chennai',lat:12.8231,lon:80.0442,label:'SRM Chennai Campus'},
  'srm trichy':{city:'Trichy',lat:10.7578,lon:78.8154,label:'SRM Trichy Campus'},
  'srm tiruchirappalli':{city:'Trichy',lat:10.7578,lon:78.8154,label:'SRM Trichy Campus'},
  'srm ncr':{city:'Delhi NCR',lat:28.4744,lon:77.5040,label:'SRM NCR Campus (Greater Noida)'},
  'srm delhi':{city:'Delhi NCR',lat:28.4744,lon:77.5040,label:'SRM Delhi NCR Campus (Greater Noida)'},
  'srm delhi ncr':{city:'Delhi NCR',lat:28.4744,lon:77.5040,label:'SRM Delhi NCR Campus (Greater Noida)'},
  'srm noida':{city:'Delhi NCR',lat:28.4744,lon:77.5040,label:'SRM NCR Campus (Greater Noida)'},
  'srm greater noida':{city:'Delhi NCR',lat:28.4744,lon:77.5040,label:'SRM NCR Campus (Greater Noida)'},
  'srm andhra':{city:'Amaravati',lat:16.4434,lon:80.5942,label:'SRM AP Campus (Amaravati)'},
  'srm andhra pradesh':{city:'Amaravati',lat:16.4434,lon:80.5942,label:'SRM AP Campus (Amaravati)'},
  'srm amaravati':{city:'Amaravati',lat:16.4434,lon:80.5942,label:'SRM AP Campus (Amaravati)'},
  'srm ap':{city:'Amaravati',lat:16.4434,lon:80.5942,label:'SRM AP Campus (Amaravati)'},
  'srm sikkim':{city:'Gangtok',lat:27.3314,lon:88.6138,label:'SRM Sikkim Campus (Gangtok)'},
  'iit madras':{city:'Chennai',lat:12.9916,lon:80.2336,label:'IIT Madras (Chennai)'},
  'iit bombay':{city:'Mumbai',lat:19.1334,lon:72.9133,label:'IIT Bombay (Mumbai)'},
  'iit delhi':{city:'Delhi',lat:28.5456,lon:77.1926,label:'IIT Delhi'},
  'vit vellore':{city:'Vellore',lat:12.9692,lon:79.1559,label:'VIT Vellore'},
  'bits pilani':{city:'Pilani',lat:28.3643,lon:75.5870,label:'BITS Pilani'},
  'anna university':{city:'Chennai',lat:13.0108,lon:80.2354,label:'Anna University (Chennai)'},
  'nit trichy':{city:'Trichy',lat:10.7601,lon:78.8137,label:'NIT Trichy'},
};

export function resolveCampus(input) {
  const key = String(input || '').toLowerCase().trim().replace(/[,.\-]/g,' ').replace(/\s+/g,' ').trim();
  if (!key) return null;

  if (/srm\s*(?:university|ist|institute)?\s*,?\s*(trichy|tiruchirappalli|tiruchi)/i.test(key)) return CAMPUS_MAP['srm trichy'];
  if (/srm\s*(?:university|ist|institute)?\s*,?\s*(ncr|delhi|noida|greater\s*noida|modinagar)/i.test(key)) return CAMPUS_MAP['srm ncr'];
  if (/srm\s*(?:university|ist|institute)?\s*,?\s*(andhra|ap|amaravati|guntur)/i.test(key)) return CAMPUS_MAP['srm andhra'];
  if (/srm\s*(?:university|ist|institute)?\s*,?\s*(sikkim|gangtok)/i.test(key)) return CAMPUS_MAP['srm sikkim'];
  if (/srm\s*(?:university|ist|institute)?\s*,?\s*(chennai|kattankulathur|chengalpattu)/i.test(key)) return CAMPUS_MAP['srmist'];

  for (const [campus, info] of Object.entries(CAMPUS_MAP)) {
    if (key === campus || key.includes(campus)) return info;
  }
  if (/^srm\b/.test(key) && !key.includes('nagar')) return CAMPUS_MAP['srmist'];
  return null;
}

export function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const toRad = (x) => (x * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1), dLon = toRad(lon2 - lon1);
  const a = Math.sin(dLat/2)**2 + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon/2)**2;
  return Math.round(2 * R * Math.asin(Math.sqrt(a)));
}

export function lookupCoords(name) {
  const key = String(name || '').toLowerCase().trim().replace(/[,.\-]/g,' ').replace(/\s+/g,' ').trim();
  const campus = CAMPUS_MAP[key];
  if (campus) return [campus.lat, campus.lon];
  if (CITY_COORDS[key]) return CITY_COORDS[key];
  for (const [c, coord] of Object.entries(CITY_COORDS)) {
    if (key.includes(c) || c.includes(key)) return coord;
  }
  return null;
}

export function getDistance(origin, dest) {
  const oKey = String(origin || '').toLowerCase().replace(/[^a-z]/g,'');
  const dKey = String(dest || '').toLowerCase().replace(/[^a-z]/g,'');
  for (const [city, dists] of Object.entries(CITY_DISTANCES)) {
    if (oKey.includes(city) || city.includes(oKey)) {
      for (const [d, km] of Object.entries(dists)) {
        if (dKey.includes(d) || d.includes(dKey)) return km;
      }
    }
  }
  const oCoords = lookupCoords(origin), dCoords = lookupCoords(dest);
  if (oCoords && dCoords) return haversineKm(oCoords[0], oCoords[1], dCoords[0], dCoords[1]);
  return 800;
}

export function getIATA(city) {
  if (!city) return '';
  const k = String(city).toLowerCase().replace(/[^a-z\s]/g,'').trim();
  for (const [name, code] of Object.entries(CITY_IATA)) {
    if (k.includes(name.replace(/_/g,' ')) || name.includes(k)) return code;
  }
  return '';
}

/* Lightweight geocoder that doesn't require a network call:
   1. campus map → 2. SRM-keyword guard → 3. CITY_COORDS substring → 4. fallback. */
export function geocode(place) {
  const key = String(place || '').toLowerCase().trim().replace(/[,.\-]/g,' ').replace(/\s+/g,' ').trim();
  if (!key) return { lat: 20.5937, lon: 78.9629, name: place || 'India', resolvedCity: place || 'India' };

  const campus = resolveCampus(key);
  if (campus) return { lat: campus.lat, lon: campus.lon, name: campus.label, resolvedCity: campus.city };

  // SRM-keyword guard: if the user typed "Chennai SRM", "Chennai SRMIST", etc.
  // the campus resolver above usually catches it via regex, but if it didn't
  // (e.g. unusual phrasing), fall back to Kattankulathur instead of dropping
  // the user at the Chennai city centre 40 km away.
  if (/\bsrm(ist)?\b/i.test(key)) {
    if (/\b(trichy|tiruchirappalli)\b/.test(key)) return { lat: 10.7578, lon: 78.8154, name: 'SRM Trichy Campus', resolvedCity: 'Trichy' };
    if (/\b(ncr|noida|delhi|gurgaon|modinagar)\b/.test(key)) return { lat: 28.4744, lon: 77.5040, name: 'SRM NCR Campus (Greater Noida)', resolvedCity: 'Delhi NCR' };
    if (/\b(andhra|amaravati|ap|guntur|vijayawada)\b/.test(key)) return { lat: 16.4434, lon: 80.5942, name: 'SRM AP Campus (Amaravati)', resolvedCity: 'Amaravati' };
    if (/\b(sikkim|gangtok)\b/.test(key)) return { lat: 27.3314, lon: 88.6138, name: 'SRM Sikkim Campus (Gangtok)', resolvedCity: 'Gangtok' };
    return { lat: 12.8231, lon: 80.0442, name: 'SRMIST Kattankulathur (Chennai)', resolvedCity: 'Chennai SRM' };
  }

  const sortedCities = Object.entries(CITY_COORDS).sort((a,b) => b[0].length - a[0].length);
  for (const [city, [lat,lon]] of sortedCities) {
    if (key.includes(city) || city.includes(key)) {
      return { lat, lon, name: place, resolvedCity: city };
    }
  }
  return { lat: 20.5937, lon: 78.9629, name: place, resolvedCity: place };
}

/* Pull the right top-attractions list for a city / SRM campus.
   If `nearLat`/`nearLon` are supplied, the list is sorted by distance to that
   anchor so itineraries built from a precise destination (e.g. "Chennai SRM")
   surface nearby POIs first instead of dropping markers all over the map. */
export function getTopAttractions(city, nearLat, nearLon) {
  const key = String(city || '').toLowerCase().trim();
  if (!key) return [];

  // Special-case SRM Kattankulathur: use the curated SRM-area list.
  if (/srm/.test(key) || (key.includes('chennai') && key.includes('srm')) ||
      key.includes('kattankulathur') || key.includes('katankulathur')) {
    const list = CITY_TOP_ATTRACTIONS['chennai srm'];
    if (list) return [...list];
  }

  let list = [];
  // Prefer exact-key match before substring fallback so 'agra' doesn't hit
  // 'agartala' style cases.
  if (CITY_TOP_ATTRACTIONS[key]) list = CITY_TOP_ATTRACTIONS[key];
  else {
    for (const [c, items] of Object.entries(CITY_TOP_ATTRACTIONS)) {
      if (c === 'chennai srm') continue; // handled above
      if (key === c || key.includes(c) || c.includes(key)) { list = items; break; }
    }
  }

  if (!list.length) return [];
  if (typeof nearLat === 'number' && typeof nearLon === 'number') {
    return [...list].sort((a, b) =>
      haversineKm(a.lat, a.lon, nearLat, nearLon) -
      haversineKm(b.lat, b.lon, nearLat, nearLon)
    );
  }
  return [...list];
}
