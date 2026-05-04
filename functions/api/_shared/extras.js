/* ════════════════════════════════════════════════════════════════
   _shared/extras.js — Restaurants, language tips, packing list,
   emergency contacts, safety tips, recommendations.
   Merged from GitHub src/index.tsx.
   ════════════════════════════════════════════════════════════════ */

export const CITY_RESTAURANTS = {
  chennai: [
    {name:'Saravana Bhavan (T. Nagar)',cuisine:'South Indian',rating:4.4,price_range:'₹₹',avgCost:300,lat:13.0418,lon:80.2341,zomato:'https://www.zomato.com/chennai/hotel-saravana-bhavan-t-nagar'},
    {name:'Murugan Idli Shop (Besant Nagar)',cuisine:'South Indian',rating:4.5,price_range:'₹',avgCost:200,lat:13.0006,lon:80.2680,zomato:'https://www.zomato.com/chennai/sri-murugan-idli-shop-besant-nagar'},
    {name:'Buhari Hotel (Anna Salai)',cuisine:'Biryani',rating:4.2,price_range:'₹₹',avgCost:500,lat:13.0608,lon:80.2566,zomato:'https://www.zomato.com/chennai/buhari-hotel-anna-salai'},
    {name:'Junior Kuppanna (Adyar)',cuisine:'Chettinad',rating:4.3,price_range:'₹₹',avgCost:600,lat:13.0067,lon:80.2566,zomato:'https://www.zomato.com/chennai/junior-kuppanna-adyar'},
    {name:'Anjappar (Nungambakkam)',cuisine:'Chettinad',rating:4.4,price_range:'₹₹',avgCost:700,lat:13.0596,lon:80.2421,zomato:'https://www.zomato.com/chennai/anjappar-chettinad-restaurant-nungambakkam'},
    {name:'Sangeetha Veg (T. Nagar)',cuisine:'South Indian',rating:4.2,price_range:'₹₹',avgCost:400,lat:13.0418,lon:80.2341,zomato:'https://www.zomato.com/chennai/sangeetha-veg-restaurant-t-nagar'},
    {name:'Mathsya (Egmore)',cuisine:'Pure Veg',rating:4.3,price_range:'₹',avgCost:250,lat:13.0732,lon:80.2609,zomato:'https://www.zomato.com/chennai/mathsya-egmore'},
  ],
  delhi: [
    {name:"Karim's (Jama Masjid)",cuisine:'Mughlai',rating:4.4,price_range:'₹₹',avgCost:600,lat:28.6489,lon:77.2356,zomato:'https://www.zomato.com/ncr/karims-jama-masjid-new-delhi'},
    {name:'Paranthe Wali Gali (Chandni Chowk)',cuisine:'Street Food',rating:4.2,price_range:'₹',avgCost:200,lat:28.6562,lon:77.2308,zomato:'https://www.zomato.com/ncr/paranthe-wali-gali-chandni-chowk-new-delhi'},
    {name:'Bukhara (ITC Maurya)',cuisine:'North Indian',rating:4.7,price_range:'₹₹₹₹',avgCost:5000,lat:28.5994,lon:77.1772,zomato:'https://www.zomato.com/ncr/bukhara-itc-maurya-diplomatic-enclave-new-delhi'},
    {name:'Saravana Bhavan (CP)',cuisine:'South Indian',rating:4.3,price_range:'₹₹',avgCost:400,lat:28.6315,lon:77.2167,zomato:'https://www.zomato.com/ncr/saravana-bhavan-connaught-place-cp-new-delhi'},
    {name:'Indian Accent (The Lodhi)',cuisine:'Modern Indian',rating:4.8,price_range:'₹₹₹₹',avgCost:5500,lat:28.5896,lon:77.2299,zomato:'https://www.zomato.com/ncr/indian-accent-the-lodhi-new-delhi'},
  ],
  mumbai: [
    {name:'Bademiya (Colaba)',cuisine:'Street Food',rating:4.3,price_range:'₹₹',avgCost:500,lat:18.9196,lon:72.8311,zomato:'https://www.zomato.com/mumbai/bademiya-colaba'},
    {name:'Britannia & Co. (Ballard Estate)',cuisine:'Parsi',rating:4.6,price_range:'₹₹₹',avgCost:1200,lat:18.9357,lon:72.8400,zomato:'https://www.zomato.com/mumbai/britannia-co-ballard-estate'},
    {name:'Trishna (Fort)',cuisine:'Seafood',rating:4.5,price_range:'₹₹₹₹',avgCost:3000,lat:18.9322,lon:72.8331,zomato:'https://www.zomato.com/mumbai/trishna-fort'},
    {name:'Leopold Cafe (Colaba)',cuisine:'Continental',rating:4.1,price_range:'₹₹',avgCost:900,lat:18.9220,lon:72.8312,zomato:'https://www.zomato.com/mumbai/leopold-cafe-bar-colaba'},
    {name:'Bombay Canteen (Lower Parel)',cuisine:'Modern Indian',rating:4.5,price_range:'₹₹₹₹',avgCost:2500,lat:18.9929,lon:72.8267,zomato:'https://www.zomato.com/mumbai/the-bombay-canteen-lower-parel'},
  ],
  bangalore: [
    {name:'MTR (Lalbagh)',cuisine:'South Indian',rating:4.5,price_range:'₹₹',avgCost:400,lat:12.9561,lon:77.5848,zomato:'https://www.zomato.com/bangalore/mtr-lalbagh'},
    {name:'Vidyarthi Bhavan (Basavanagudi)',cuisine:'South Indian',rating:4.4,price_range:'₹',avgCost:200,lat:12.9408,lon:77.5728,zomato:'https://www.zomato.com/bangalore/vidyarthi-bhavan-basavanagudi'},
    {name:'Truffles (Koramangala)',cuisine:'American',rating:4.6,price_range:'₹₹₹',avgCost:900,lat:12.9352,lon:77.6245,zomato:'https://www.zomato.com/bangalore/truffles-koramangala'},
    {name:'Karavalli (Taj Gateway)',cuisine:'Coastal',rating:4.6,price_range:'₹₹₹₹',avgCost:3500,lat:12.9590,lon:77.5970,zomato:'https://www.zomato.com/bangalore/karavalli-residency-road'},
  ],
  jaipur: [
    {name:'Laxmi Mishthan Bhandar (LMB)',cuisine:'Rajasthani',rating:4.4,price_range:'₹₹',avgCost:500,lat:26.9213,lon:75.8267,zomato:'https://www.zomato.com/jaipur/laxmi-misthan-bhandar-lmb-johari-bazaar'},
    {name:'Chokhi Dhani',cuisine:'Rajasthani',rating:4.5,price_range:'₹₹₹',avgCost:1500,lat:26.7681,lon:75.7998,zomato:'https://www.zomato.com/jaipur/chokhi-dhani-tonk-road'},
    {name:'Suvarna Mahal (Rambagh Palace)',cuisine:'Royal Indian',rating:4.7,price_range:'₹₹₹₹',avgCost:5000,lat:26.8911,lon:75.8077,zomato:'https://www.zomato.com/jaipur/suvarna-mahal-rambagh-palace'},
    {name:'Rawat Mishtan Bhandar',cuisine:'Rajasthani Sweets',rating:4.3,price_range:'₹',avgCost:200,lat:26.9216,lon:75.7915,zomato:'https://www.zomato.com/jaipur/rawat-mishtan-bhandar-sindhi-camp'},
  ],
  goa: [
    {name:"Britto's (Baga)",cuisine:'Goan',rating:4.2,price_range:'₹₹₹',avgCost:1500,lat:15.5550,lon:73.7510,zomato:'https://www.zomato.com/goa/brittos-baga'},
    {name:"Fisherman's Wharf (Cavelossim)",cuisine:'Goan Seafood',rating:4.4,price_range:'₹₹₹',avgCost:1500,lat:15.1740,lon:73.9410,zomato:'https://www.zomato.com/goa/fishermans-wharf-cavelossim'},
    {name:'Souza Lobo (Calangute)',cuisine:'Goan',rating:4.3,price_range:'₹₹₹',avgCost:1500,lat:15.5440,lon:73.7530,zomato:'https://www.zomato.com/goa/souza-lobo-calangute'},
  ],
  hyderabad: [
    {name:'Paradise Biryani (Secunderabad)',cuisine:'Biryani',rating:4.3,price_range:'₹₹',avgCost:600,lat:17.4399,lon:78.4983,zomato:'https://www.zomato.com/hyderabad/paradise-secunderabad'},
    {name:'Bawarchi (RTC X Roads)',cuisine:'Biryani',rating:4.4,price_range:'₹₹',avgCost:500,lat:17.4072,lon:78.4986,zomato:'https://www.zomato.com/hyderabad/bawarchi-rtc-x-roads'},
    {name:'Shah Ghouse (Tolichowki)',cuisine:'Hyderabadi',rating:4.3,price_range:'₹₹',avgCost:600,lat:17.3939,lon:78.4090,zomato:'https://www.zomato.com/hyderabad/shah-ghouse-cafe-restaurant-tolichowki'},
  ],
  kolkata: [
    {name:'Peter Cat (Park Street)',cuisine:'Continental',rating:4.4,price_range:'₹₹₹',avgCost:1200,lat:22.5520,lon:88.3520,zomato:'https://www.zomato.com/kolkata/peter-cat-park-street-area'},
    {name:'Arsalan (Park Circus)',cuisine:'Mughlai',rating:4.3,price_range:'₹₹',avgCost:700,lat:22.5410,lon:88.3700,zomato:'https://www.zomato.com/kolkata/arsalan-park-circus-area'},
    {name:'Bhojohori Manna (Ekdalia)',cuisine:'Bengali',rating:4.2,price_range:'₹₹',avgCost:600,lat:22.5230,lon:88.3700,zomato:'https://www.zomato.com/kolkata/bhojohori-manna-ekdalia'},
  ],
  pondicherry: [
    {name:'Cafe des Arts',cuisine:'French',rating:4.4,price_range:'₹₹',avgCost:600,lat:11.9340,lon:79.8370,zomato:'https://www.zomato.com/pondicherry/cafe-des-arts-white-town'},
    {name:'La Pasta',cuisine:'Italian',rating:4.5,price_range:'₹₹₹',avgCost:1200,lat:11.9355,lon:79.8365,zomato:'https://www.zomato.com/pondicherry/la-pasta-white-town'},
    {name:'Surguru Restaurant',cuisine:'South Indian',rating:4.2,price_range:'₹₹',avgCost:400,lat:11.9416,lon:79.8083,zomato:'https://www.zomato.com/pondicherry/surguru-mission-street'},
  ],
  trichy: [
    {name:'Hotel Sangam (Thillai Nagar)',cuisine:'South Indian',rating:4.3,price_range:'₹₹',avgCost:500,lat:10.8155,lon:78.6913,zomato:'https://www.zomato.com/trichy/hotel-sangam-thillai-nagar'},
    {name:'Banana Leaf (Cantonment)',cuisine:'South Indian',rating:4.2,price_range:'₹₹',avgCost:400,lat:10.8155,lon:78.6913,zomato:'https://www.zomato.com/trichy/banana-leaf-cantonment'},
    {name:'Vasanta Bhavan',cuisine:'South Indian',rating:4.3,price_range:'₹',avgCost:250,lat:10.8085,lon:78.6946,zomato:'https://www.zomato.com/trichy/vasanta-bhavan-thillai-nagar'},
  ],
  agra: [
    {name:'Pinch of Spice (Tajganj)',cuisine:'North Indian',rating:4.4,price_range:'₹₹₹',avgCost:1300,lat:27.1605,lon:78.0410,zomato:'https://www.zomato.com/agra/pinch-of-spice-tajganj'},
    {name:'Esphahan (Oberoi Amarvilas)',cuisine:'North Indian',rating:4.7,price_range:'₹₹₹₹',avgCost:5000,lat:27.1605,lon:78.0490,zomato:'https://www.zomato.com/agra/esphahan-the-oberoi-amarvilas-tajganj'},
    {name:'Shankara Vegis Restaurant',cuisine:'Vegetarian',rating:4.2,price_range:'₹₹',avgCost:500,lat:27.1700,lon:78.0420,zomato:'https://www.zomato.com/agra/shankara-vegis-restaurant-tajganj'},
  ],
};

export function generateRestaurants(city, lat, lon) {
  const cityKey = String(city || '').toLowerCase().replace(/[^a-z\s]/g,'').trim();
  let real = [];
  for (const [k, v] of Object.entries(CITY_RESTAURANTS)) {
    if (cityKey.includes(k) || k.includes(cityKey)) { real = v; break; }
  }
  if (real.length) {
    return real.map((r, i) => ({
      id: `RS${i}`,
      name: r.name, cuisine: r.cuisine, rating: r.rating.toFixed(1),
      price_range: r.price_range, avgCost: r.avgCost,
      lat: r.lat, lon: r.lon,
      bookingUrl: r.zomato,
      mapsUrl: `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(r.name + ' ' + city)}`,
    }));
  }
  const cuisines = ['South Indian','North Indian','Chinese','Continental','Street Food','Biryani'];
  const seed = (cityKey.length || 7);
  return Array.from({length:6},(_,i) => {
    const cost = 200 + ((seed * 13 + i * 47) % 700);
    const rating = (3.7 + ((seed + i * 3) % 13) / 10).toFixed(1);
    const priceRanges = ['₹','₹₹','₹₹₹'];
    return {
      id: `RS${i}`,
      name: `${['Spice','Royal','Golden','Green','Silver','Paradise'][i]} ${['Kitchen','Restaurant','Diner','Cafe','Palace','Garden'][i]} (${city})`,
      cuisine: cuisines[i % cuisines.length], rating,
      price_range: priceRanges[i % 3], avgCost: cost,
      lat: (lat||13) + ((i % 3 - 1) * 0.005), lon: (lon||80) + ((i % 5 - 2) * 0.005),
      bookingUrl: `https://www.zomato.com/${String(city||'').toLowerCase().replace(/\s+/g,'-')}`,
      mapsUrl: `https://www.google.com/maps/search/?api=1&query=restaurants+near+${encodeURIComponent(city||'')}`,
    };
  });
}

export function getLanguageTips(city) {
  const regionMap = {
    chennai: {language:'Tamil',phrases:[{phrase:'Vanakkam',meaning:'Hello',pronunciation:'va-NAK-kam'},{phrase:'Nandri',meaning:'Thank You',pronunciation:'NAN-dri'},{phrase:'Evvalavu?',meaning:'How much?',pronunciation:'ev-va-LA-vu'},{phrase:'Sapadu',meaning:'Food',pronunciation:'SAA-pa-du'},{phrase:'Thanni',meaning:'Water',pronunciation:'THAN-ni'},{phrase:'Illa',meaning:'No',pronunciation:'IL-la'},{phrase:'Aamaa',meaning:'Yes',pronunciation:'AA-maa'}]},
    mumbai: {language:'Hindi/Marathi',phrases:[{phrase:'Namaste',meaning:'Hello',pronunciation:'na-MAS-tay'},{phrase:'Dhanyavaad',meaning:'Thank You',pronunciation:'dhan-ya-VAAD'},{phrase:'Kitna?',meaning:'How much?',pronunciation:'KIT-na'},{phrase:'Khaana',meaning:'Food',pronunciation:'KHAA-na'},{phrase:'Paani',meaning:'Water',pronunciation:'PAA-ni'}]},
    jaipur: {language:'Hindi/Rajasthani',phrases:[{phrase:'Khamma Ghani',meaning:'Hello (Rajasthani)',pronunciation:'KHAM-ma GHA-ni'},{phrase:'Shukriya',meaning:'Thank You',pronunciation:'shuk-RI-ya'},{phrase:'Kitna hai?',meaning:'How much?',pronunciation:'KIT-na hai'}]},
    delhi: {language:'Hindi',phrases:[{phrase:'Namaste',meaning:'Hello',pronunciation:'na-MAS-tay'},{phrase:'Shukriya',meaning:'Thank You',pronunciation:'shuk-RI-ya'},{phrase:'Kidhar hai?',meaning:'Where is it?',pronunciation:'KID-har hai'},{phrase:'Kitne ka hai?',meaning:'How much?',pronunciation:'KIT-ne ka hai'}]},
    kolkata: {language:'Bengali',phrases:[{phrase:'Nomoshkar',meaning:'Hello',pronunciation:'no-mosh-KAR'},{phrase:'Dhonnobad',meaning:'Thank You',pronunciation:'dhon-no-BAD'},{phrase:'Koto dam?',meaning:'How much?',pronunciation:'ko-to DAM'}]},
    bangalore: {language:'Kannada',phrases:[{phrase:'Namaskara',meaning:'Hello',pronunciation:'na-mas-KA-ra'},{phrase:'Dhanyavadagalu',meaning:'Thank You',pronunciation:'dhan-ya-VA-da-ga-lu'},{phrase:'Eshthu?',meaning:'How much?',pronunciation:'ESH-thu'}]},
    hyderabad: {language:'Telugu/Urdu',phrases:[{phrase:'Namaskaaram',meaning:'Hello',pronunciation:'na-mas-KAA-ram'},{phrase:'Dhanyavaadaalu',meaning:'Thank You',pronunciation:'dhan-ya-VAA-daa-lu'}]},
    kochi: {language:'Malayalam',phrases:[{phrase:'Namaskaram',meaning:'Hello',pronunciation:'na-mas-KA-ram'},{phrase:'Nanni',meaning:'Thank You',pronunciation:'NAN-ni'},{phrase:'Ethra?',meaning:'How much?',pronunciation:'ETH-ra'}]},
    trichy: {language:'Tamil',phrases:[{phrase:'Vanakkam',meaning:'Hello',pronunciation:'va-NAK-kam'},{phrase:'Nandri',meaning:'Thank You',pronunciation:'NAN-dri'}]},
    amaravati: {language:'Telugu',phrases:[{phrase:'Namaskaaram',meaning:'Hello',pronunciation:'na-mas-KAA-ram'},{phrase:'Dhanyavaadaalu',meaning:'Thank You',pronunciation:'dhan-ya-VAA-daa-lu'}]},
  };
  const key = String(city||'').toLowerCase().replace(/[^a-z]/g,'');
  for (const [c, data] of Object.entries(regionMap)) { if (key.includes(c) || c.includes(key)) return data; }
  return {language:'Hindi (default)',phrases:[{phrase:'Namaste',meaning:'Hello',pronunciation:'na-MAS-tay'},{phrase:'Dhanyavaad',meaning:'Thank You',pronunciation:'dhan-ya-VAAD'},{phrase:'Kitna?',meaning:'How much?',pronunciation:'KIT-na'},{phrase:'Khaana',meaning:'Food',pronunciation:'KHAA-na'},{phrase:'Paani',meaning:'Water',pronunciation:'PAA-ni'},{phrase:'Haan',meaning:'Yes',pronunciation:'HAAN'},{phrase:'Nahi',meaning:'No',pronunciation:'na-HI'},{phrase:'Madat',meaning:'Help',pronunciation:'MA-dat'}]};
}

export function generatePackingList(days, weather, persona) {
  const categories = {
    'Essentials': ['Passport/ID','Phone + Charger','Power Bank','Cash + Cards','Travel Insurance Docs','Medicines'],
    'Clothing': [`${days+1} T-shirts/Tops`,`${days} Pants/Shorts`,'Comfortable Walking Shoes','Sleepwear','Undergarments'],
    'Toiletries': ['Toothbrush + Paste','Sunscreen SPF 50','Deodorant','Hand Sanitizer','Wet Wipes','Lip Balm'],
    'Tech': ['Phone Charger','Earphones','Camera (optional)','Universal Adapter'],
    'Travel Comfort': ['Neck Pillow','Eye Mask','Reusable Water Bottle','Snacks'],
  };
  const w = weather || [];
  const hasRain = w.some(x => x.risk_level === 'high' || x.precipitation > 5);
  const hasHeat = w.some(x => x.temp_max > 35);
  const hasCold = w.some(x => x.temp_min < 15);

  if (hasRain) categories['Weather Prep'] = ['Umbrella/Raincoat','Waterproof Bag','Quick-dry Towel'];
  if (hasHeat) { categories['Clothing'].push('Hat/Cap','Sunglasses'); categories['Toiletries'].push('After-sun Lotion'); }
  if (hasCold) categories['Clothing'].push('Jacket/Sweater','Warm Socks','Gloves');
  if (persona === 'adventure') categories['Adventure Gear'] = ['Hiking Boots','Daypack','First Aid Kit','Torch/Headlamp','Compass','Insect Repellent'];
  if (persona === 'luxury')    categories['Luxury'] = ['Formal Outfit','Jewelry','Premium Toiletry Kit','Travel Pillow (Memory Foam)'];
  if (persona === 'family')    categories['Family Essentials'] = ['Kids Snacks','Entertainment for Children','First Aid Kit','Baby Wipes','Extra Bags'];

  return categories;
}

export function getEmergencyContacts(city) {
  const base = {
    police: '100', ambulance: '108', fire: '101',
    women_helpline: '1091', tourist_helpline: '1363',
    disaster_mgmt: '1078', universal: '112',
    roadside_assistance: '1033'
  };
  const citySpecific = {
    chennai:   { ...base, local_police: '044-28447777',   hospital: 'Apollo Hospital: 044-28290200',   tourist_office: '044-25340802' },
    mumbai:    { ...base, local_police: '022-22621855',   hospital: 'Lilavati Hospital: 022-26751000',  tourist_office: '022-22074333' },
    delhi:     { ...base, local_police: '011-23490100',   hospital: 'AIIMS: 011-26588500',              embassy: 'US Embassy: 011-24198000', tourist_office: '011-23365358' },
    jaipur:    { ...base, local_police: '0141-2560063',   hospital: 'SMS Hospital: 0141-2518291',       tourist_office: '0141-5110598' },
    goa:       { ...base, local_police: '0832-2225003',   hospital: 'GMC Hospital: 0832-2458727',       tourist_office: '0832-2438750' },
    bangalore: { ...base, local_police: '080-22942222',   hospital: 'Manipal Hospital: 080-25024444',   tourist_office: '080-22352828' },
    kolkata:   { ...base, local_police: '033-22145050',   hospital: 'AMRI Hospital: 033-66261000',      tourist_office: '033-22485917' },
    hyderabad: { ...base, local_police: '040-27852400',   hospital: 'NIMS: 040-23390631',                tourist_office: '040-23262143' },
  };
  const key = String(city||'').toLowerCase().replace(/[^a-z]/g,'');
  for (const [c, data] of Object.entries(citySpecific)) { if (key.includes(c) || c.includes(key)) return data; }
  return base;
}

export function getSafetyTips(city, persona) {
  const general = [
    'Keep copies of all documents (digital + physical)',
    'Share your itinerary with family/friends',
    'Use registered taxis/cabs only',
    'Keep emergency numbers handy (Universal: 112)',
    'Stay in well-lit areas at night',
    'Use hotel safes for valuables and extra cash',
    'Download offline maps for the destination',
    'Carry a basic first aid kit',
    'Stay hydrated and carry a water bottle',
    'Be aware of local scams and tourist traps',
  ];
  const cityTips = {
    delhi: ['Metro is safest public transport','Avoid auto-rickshaws without meters','Prepaid taxi counters at airport/station'],
    mumbai: ['Use local trains during non-peak hours','Carry change for local transport','Avoid lonely beaches at night'],
    jaipur: ['Negotiate prices at markets','Carry water in summer (40°C+)','Beware of "guide" scams at forts'],
    goa: ['Rent two-wheelers with proper license','Do NOT swim at unmarked beaches','Keep valuables secure on beaches'],
    varanasi: ['Wear comfortable shoes for ghats','Bargain for boat rides','Be cautious of self-appointed guides'],
  };
  const personaTips = {
    solo: ['Stay in hostels to meet other travelers','Share your live location with someone','Trust your instincts in unfamiliar areas'],
    family: ['Plan kid-friendly activities','Carry entertainment for children during travel','Book family rooms in advance'],
    adventure: ['Check equipment before adventure activities','Hire certified guides for treks','Carry emergency supplies'],
    luxury: ['Book premium lounge access at airports','Pre-arrange airport transfers','Verify hotel cancellation policies'],
  };
  const key = String(city||'').toLowerCase().replace(/[^a-z]/g,'');
  const extra = [];
  for (const [c, tips] of Object.entries(cityTips)) { if (key.includes(c)) extra.push(...tips); }
  return [...general, ...extra, ...(personaTips[persona]||[])];
}

export function getRecommendations(budget, duration, preferences, currentLocation) {
  const destinations = [
    {name:'Jaipur',state:'Rajasthan',tags:['culture','history','shopping','food'],budget_range:[8000,25000],best_months:['october','november','december','january','february','march'],weather:'warm',coords:[26.91,75.79],highlights:['Amber Fort','Hawa Mahal','City Palace','Nahargarh Fort']},
    {name:'Goa',state:'Goa',tags:['beach','nightlife','food','adventure'],budget_range:[10000,40000],best_months:['november','december','january','february','march'],weather:'warm',coords:[15.30,74.12],highlights:['Baga Beach','Fort Aguada','Dudhsagar Falls']},
    {name:'Manali',state:'Himachal Pradesh',tags:['adventure','nature','spiritual'],budget_range:[8000,30000],best_months:['march','april','may','june','september','october'],weather:'cold',coords:[32.24,77.19],highlights:['Rohtang Pass','Solang Valley','Old Manali']},
    {name:'Varanasi',state:'Uttar Pradesh',tags:['spiritual','culture','history','food'],budget_range:[5000,15000],best_months:['october','november','december','january','february','march'],weather:'moderate',coords:[25.32,83.01],highlights:['Dashashwamedh Ghat','Kashi Vishwanath Temple','Sarnath']},
    {name:'Udaipur',state:'Rajasthan',tags:['culture','history','nature'],budget_range:[8000,25000],best_months:['september','october','november','december','january','february','march'],weather:'moderate',coords:[24.59,73.71],highlights:['City Palace','Lake Pichola','Jag Mandir']},
    {name:'Pondicherry',state:'Tamil Nadu',tags:['beach','culture','food','history'],budget_range:[5000,20000],best_months:['october','november','december','january','february','march'],weather:'warm',coords:[11.94,79.81],highlights:['Promenade Beach','Auroville','French Quarter']},
    {name:'Darjeeling',state:'West Bengal',tags:['nature','adventure','food'],budget_range:[7000,20000],best_months:['march','april','may','september','october','november'],weather:'cold',coords:[27.04,88.27],highlights:['Tiger Hill','Toy Train','Tea Gardens']},
    {name:'Munnar',state:'Kerala',tags:['nature','adventure'],budget_range:[6000,18000],best_months:['september','october','november','december','january','february','march'],weather:'moderate',coords:[10.09,77.06],highlights:['Tea Plantations','Eravikulam National Park','Mattupetty Dam']},
    {name:'Hampi',state:'Karnataka',tags:['history','culture','adventure'],budget_range:[4000,12000],best_months:['october','november','december','january','february'],weather:'warm',coords:[15.34,76.46],highlights:['Virupaksha Temple','Vittala Temple','Royal Enclosure']},
    {name:'Alleppey',state:'Kerala',tags:['nature','food','culture'],budget_range:[8000,25000],best_months:['august','september','october','november','december','january','february','march'],weather:'warm',coords:[9.50,76.34],highlights:['Houseboat Cruise','Alappuzha Beach','Kumarakom Bird Sanctuary']},
    {name:'Rishikesh',state:'Uttarakhand',tags:['spiritual','adventure','nature'],budget_range:[5000,15000],best_months:['february','march','april','may','september','october','november'],weather:'moderate',coords:[30.09,78.27],highlights:['Ram Jhula','Rafting','Triveni Ghat']},
    {name:'Leh Ladakh',state:'Ladakh',tags:['adventure','nature'],budget_range:[15000,50000],best_months:['june','july','august','september'],weather:'cold',coords:[34.15,77.58],highlights:['Pangong Lake','Nubra Valley','Khardung La']},
    {name:'Ooty',state:'Tamil Nadu',tags:['nature','food'],budget_range:[5000,15000],best_months:['march','april','may','october','november'],weather:'cold',coords:[11.41,76.70],highlights:['Botanical Garden','Ooty Lake','Nilgiri Mountain Railway']},
    {name:'Kodaikanal',state:'Tamil Nadu',tags:['nature','adventure'],budget_range:[5000,15000],best_months:['march','april','may','september','october'],weather:'cold',coords:[10.24,77.49],highlights:['Kodai Lake','Coakers Walk','Pillar Rocks']},
    {name:'Amritsar',state:'Punjab',tags:['spiritual','food','history','culture'],budget_range:[5000,15000],best_months:['october','november','december','january','february','march'],weather:'moderate',coords:[31.63,74.87],highlights:['Golden Temple','Wagah Border','Jallianwala Bagh']},
    {name:'Mahabalipuram',state:'Tamil Nadu',tags:['beach','history','culture'],budget_range:[3000,10000],best_months:['november','december','january','february','march'],weather:'warm',coords:[12.62,80.20],highlights:["Shore Temple","Pancha Rathas","Arjuna's Penance"]},
  ];
  return destinations.filter(d => {
    if (d.budget_range[0] > budget) return false;
    if ((preferences||[]).length && !preferences.some(p => d.tags.includes(p))) return false;
    return true;
  }).map(d => ({
    ...d,
    estimatedCost: Math.round(d.budget_range[0] + (d.budget_range[1]-d.budget_range[0])*((duration||3)/7)),
    matchScore: (preferences||[]).filter(p => d.tags.includes(p)).length / Math.max((preferences||[]).length, 1) * 100,
  })).sort((a,b) => b.matchScore - a.matchScore).slice(0, 8);
}

export function compareTrips(trips) {
  if (!trips?.length) return [];
  return trips.map(t => ({
    destination: t.destination,
    days: t.days,
    totalCost: t.totalCost,
    budget: t.budget,
    budgetUtilization: Math.round((t.totalCost / Math.max(t.budget,1)) * 100),
    activitiesCount: t.days_data?.reduce((s,d) => s + (d.activities?.length||0), 0) || 0,
    avgCrowd: Math.round((t.days_data?.flatMap(d => d.activities||[]).reduce((s,a) => s + (a.crowd_level||50), 0) || 0) / Math.max(t.days_data?.flatMap(d => d.activities||[]).length||1, 1)),
    rainyDays: (t.weather||[]).filter(w => w.risk_level === 'high').length,
    weatherQuality: Math.round(((t.weather||[]).filter(w => w.risk_level !== 'high').length / Math.max((t.weather||[]).length, 1)) * 100),
  }));
}
