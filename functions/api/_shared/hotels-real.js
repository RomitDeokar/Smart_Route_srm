/* ════════════════════════════════════════════════════════════════
   _shared/hotels-real.js — Curated REAL hotel database per city,
   merged from GitHub src/index.tsx (RomitDeokar/Smart_Route_srm).

   These are real hotel names + neighborhoods + price points.
   ════════════════════════════════════════════════════════════════ */

import { isSRMCity, SRM_SPECIFIC_HOTELS } from './srm.js';

export const REAL_HOTELS_BY_CITY = {
  delhi: [
    {name:'Taj Palace, New Delhi',          area:'Diplomatic Enclave',  stars:5, basePrice:14500, rating:4.7, amenities:['WiFi','Pool','Spa','Gym','Concierge','Restaurant','Bar']},
    {name:'The Leela Palace New Delhi',     area:'Chanakyapuri',        stars:5, basePrice:18500, rating:4.8, amenities:['WiFi','Pool','Spa','Butler','Gym','Restaurant']},
    {name:'ITC Maurya, A Luxury Collection',area:'Diplomatic Enclave',  stars:5, basePrice:13800, rating:4.7, amenities:['WiFi','Pool','Spa','Gym','Bukhara Restaurant']},
    {name:'The Imperial New Delhi',         area:'Janpath',             stars:5, basePrice:12200, rating:4.6, amenities:['WiFi','Pool','Spa','Heritage Property']},
    {name:'Radisson Blu Plaza Delhi Airport',area:'Mahipalpur',         stars:5, basePrice:7800,  rating:4.4, amenities:['WiFi','Pool','Gym','Airport Shuttle']},
    {name:'Lemon Tree Premier, Delhi Airport',area:'Aerocity',          stars:4, basePrice:5400,  rating:4.3, amenities:['WiFi','Pool','Gym','Restaurant','Airport Shuttle']},
    {name:'Holiday Inn New Delhi Mayur Vihar',area:'Mayur Vihar',       stars:4, basePrice:5200,  rating:4.2, amenities:['WiFi','Pool','Gym','Restaurant']},
    {name:'Bloomrooms @ New Delhi Railway Station',area:'Paharganj',    stars:3, basePrice:2400,  rating:4.1, amenities:['WiFi','AC','Breakfast']},
    {name:'Treebo Trend Daksh',             area:'Paharganj',           stars:3, basePrice:1800,  rating:3.9, amenities:['WiFi','AC','Breakfast']},
    {name:'OYO Townhouse 084 Karol Bagh',   area:'Karol Bagh',          stars:3, basePrice:1500,  rating:3.7, amenities:['WiFi','AC','Breakfast']},
    {name:'FabHotel Prime Cosmo',           area:'Mahipalpur',          stars:3, basePrice:1700,  rating:3.8, amenities:['WiFi','AC','Restaurant','Airport Shuttle']},
    {name:'Zostel Delhi (Hostel)',          area:'Paharganj',           stars:2, basePrice:700,   rating:4.2, amenities:['WiFi','Common Room','Breakfast','Backpacker']},
  ],
  mumbai: [
    {name:'The Taj Mahal Palace, Mumbai',   area:'Colaba',              stars:5, basePrice:22500, rating:4.8, amenities:['WiFi','Pool','Spa','Heritage','Sea View']},
    {name:'The Oberoi Mumbai',              area:'Nariman Point',       stars:5, basePrice:21000, rating:4.8, amenities:['WiFi','Pool','Spa','Sea View','Butler']},
    {name:'Trident Nariman Point',          area:'Nariman Point',       stars:5, basePrice:14500, rating:4.7, amenities:['WiFi','Pool','Sea View','Gym']},
    {name:'Four Seasons Hotel Mumbai',      area:'Worli',               stars:5, basePrice:17500, rating:4.7, amenities:['WiFi','Pool','Spa','Aer Rooftop Bar']},
    {name:'JW Marriott Mumbai Juhu',        area:'Juhu',                stars:5, basePrice:13800, rating:4.6, amenities:['WiFi','Pool','Spa','Beach Access']},
    {name:'The Westin Mumbai Garden City',  area:'Goregaon East',       stars:5, basePrice:9800,  rating:4.5, amenities:['WiFi','Pool','Spa','Gym']},
    {name:'Novotel Mumbai Juhu Beach',      area:'Juhu',                stars:5, basePrice:9500,  rating:4.4, amenities:['WiFi','Pool','Gym','Beach View']},
    {name:'Lemon Tree Premier MIA',         area:'Andheri East',        stars:4, basePrice:6800,  rating:4.3, amenities:['WiFi','Pool','Gym','Airport Shuttle']},
    {name:'Treebo Trend Sea Pearl',         area:'Bandra West',         stars:3, basePrice:3200,  rating:4.0, amenities:['WiFi','AC','Breakfast']},
    {name:'OYO Townhouse Bandra',           area:'Bandra West',         stars:3, basePrice:2400,  rating:3.8, amenities:['WiFi','AC','Breakfast']},
    {name:'FabHotel Prime Tashveen',        area:'Andheri East',        stars:3, basePrice:2100,  rating:3.7, amenities:['WiFi','AC','Restaurant']},
    {name:'Zostel Mumbai (Hostel)',         area:'Andheri West',        stars:2, basePrice:850,   rating:4.3, amenities:['WiFi','Common Room','Breakfast','Backpacker']},
  ],
  bangalore: [
    {name:'The Leela Palace Bengaluru',     area:'Old Airport Road',    stars:5, basePrice:13500, rating:4.7, amenities:['WiFi','Pool','Spa','Butler','Gym']},
    {name:'ITC Gardenia',                    area:'Residency Road',      stars:5, basePrice:13800, rating:4.7, amenities:['WiFi','Pool','Spa','Gym','LEED Platinum']},
    {name:'Taj West End',                    area:'Race Course Road',    stars:5, basePrice:14500, rating:4.7, amenities:['WiFi','Pool','Spa','20-acre Heritage']},
    {name:'JW Marriott Hotel Bengaluru',     area:'Vittal Mallya Road',  stars:5, basePrice:11200, rating:4.6, amenities:['WiFi','Pool','Spa','Gym']},
    {name:'The Oberoi Bengaluru',            area:'MG Road',             stars:5, basePrice:13200, rating:4.7, amenities:['WiFi','Pool','Spa','Heritage Trees']},
    {name:'Sheraton Grand Bangalore Whitefield',area:'Whitefield',      stars:5, basePrice:8800,  rating:4.5, amenities:['WiFi','Pool','Spa','Tech Park']},
    {name:'Lemon Tree Premier, Ulsoor Lake', area:'Ulsoor',              stars:4, basePrice:5400,  rating:4.3, amenities:['WiFi','Pool','Gym','Lake View']},
    {name:'Novotel Bengaluru ORR',           area:'Sarjapur Road',       stars:4, basePrice:5800,  rating:4.4, amenities:['WiFi','Pool','Gym','Restaurant']},
    {name:'Treebo Trend Pearl Suites',       area:'Indiranagar',         stars:3, basePrice:2800,  rating:4.1, amenities:['WiFi','AC','Breakfast']},
    {name:'OYO Townhouse 029 Koramangala',   area:'Koramangala',         stars:3, basePrice:2100,  rating:3.9, amenities:['WiFi','AC','Breakfast']},
    {name:'FabHotel Prime The President',    area:'MG Road',             stars:3, basePrice:1900,  rating:3.8, amenities:['WiFi','AC','Restaurant']},
    {name:'Zostel Bangalore (Hostel)',       area:'Indiranagar',         stars:2, basePrice:750,   rating:4.4, amenities:['WiFi','Common Room','Breakfast','Backpacker']},
  ],
  jaipur: [
    {name:'Rambagh Palace, Jaipur (Taj)',    area:'Bhawani Singh Road',  stars:5, basePrice:34500, rating:4.9, amenities:['WiFi','Pool','Heritage Palace','Royal Suite']},
    {name:'The Oberoi Rajvilas',             area:'Goner Road',          stars:5, basePrice:28800, rating:4.9, amenities:['WiFi','Pool','Spa','Tented Villas','32-acre']},
    {name:'Jai Mahal Palace, Jaipur (Taj)',  area:'Civil Lines',         stars:5, basePrice:13800, rating:4.7, amenities:['WiFi','Pool','Heritage','Mughal Gardens']},
    {name:'ITC Rajputana',                    area:'Palace Road',         stars:5, basePrice:11200, rating:4.6, amenities:['WiFi','Pool','Spa','Rajputana Architecture']},
    {name:'Trident Jaipur',                   area:'Amer Road',           stars:5, basePrice:9800,  rating:4.5, amenities:['WiFi','Pool','Lake View','Spa']},
    {name:'Radisson Blu Jaipur',              area:'Tonk Road',           stars:5, basePrice:7800,  rating:4.4, amenities:['WiFi','Pool','Gym','Spa']},
    {name:'Lemon Tree Premier, Jaipur',       area:'Tonk Road',           stars:4, basePrice:5200,  rating:4.3, amenities:['WiFi','Pool','Gym','Restaurant']},
    {name:'Treebo Trend Hari Mahal Palace',   area:'Bani Park',           stars:3, basePrice:2500,  rating:4.0, amenities:['WiFi','AC','Breakfast','Heritage']},
    {name:'OYO Flagship Pink City',           area:'MI Road',             stars:3, basePrice:1700,  rating:3.8, amenities:['WiFi','AC','Breakfast']},
    {name:'Zostel Jaipur (Hostel)',           area:'Bani Park',           stars:2, basePrice:650,   rating:4.4, amenities:['WiFi','Rooftop','Breakfast','Backpacker']},
  ],
  goa: [
    {name:'Taj Exotica Resort & Spa Goa',     area:'Benaulim',            stars:5, basePrice:18500, rating:4.7, amenities:['WiFi','Beach','Pool','Spa','56-acre Resort']},
    {name:'The Leela Goa',                     area:'Cavelossim',          stars:5, basePrice:17800, rating:4.7, amenities:['WiFi','Beach','Pool','Spa','Casino']},
    {name:'Park Hyatt Goa Resort & Spa',       area:'Arossim',             stars:5, basePrice:15500, rating:4.7, amenities:['WiFi','Beach','Pool','Sereno Spa']},
    {name:'W Goa',                              area:'Vagator',             stars:5, basePrice:14800, rating:4.6, amenities:['WiFi','Beach','Pool','Beach Club']},
    {name:'Caravela Beach Resort',              area:'Varca',               stars:5, basePrice:9800,  rating:4.5, amenities:['WiFi','Beach','Pool','Golf']},
    {name:'Novotel Goa Resort & Spa',           area:'Candolim',            stars:5, basePrice:8500,  rating:4.4, amenities:['WiFi','Pool','Spa','Beach Shuttle']},
    {name:'Lemon Tree Amarante Beach Resort',   area:'Candolim',            stars:4, basePrice:5800,  rating:4.3, amenities:['WiFi','Pool','Gym','Beach Access']},
    {name:'Treebo Trend Apollo Bay',            area:'Calangute',           stars:3, basePrice:2900,  rating:4.0, amenities:['WiFi','Pool','AC','Breakfast']},
    {name:'OYO Townhouse Baga Beach',           area:'Baga',                stars:3, basePrice:2200,  rating:3.8, amenities:['WiFi','AC','Breakfast']},
    {name:'Zostel Goa (Hostel)',                area:'Anjuna',              stars:2, basePrice:850,   rating:4.5, amenities:['WiFi','Pool','Beach','Backpacker']},
  ],
  chennai: [
    {name:'ITC Grand Chola, Chennai',           area:'Guindy',              stars:5, basePrice:13500, rating:4.7, amenities:['WiFi','Pool','Spa','LEED Platinum']},
    {name:'Taj Coromandel',                      area:'Nungambakkam',        stars:5, basePrice:11800, rating:4.7, amenities:['WiFi','Pool','Spa','Southern Spice']},
    {name:'The Leela Palace Chennai',            area:'MRC Nagar',           stars:5, basePrice:13200, rating:4.7, amenities:['WiFi','Pool','Spa','Sea View']},
    {name:'Hyatt Regency Chennai',               area:'Mount Road',          stars:5, basePrice:8800,  rating:4.5, amenities:['WiFi','Pool','Spa','Restaurant']},
    {name:'Novotel Chennai OMR',                  area:'OMR (Sholinganallur)',stars:4, basePrice:5800,  rating:4.4, amenities:['WiFi','Pool','Gym','IT Corridor']},
    {name:'Lemon Tree Premier, Chennai',          area:'OMR',                 stars:4, basePrice:5200,  rating:4.3, amenities:['WiFi','Pool','Gym','Restaurant']},
    {name:'Treebo Trend Adyar Gate',              area:'Adyar',               stars:3, basePrice:2400,  rating:4.0, amenities:['WiFi','AC','Breakfast']},
    {name:'OYO Townhouse 077 T Nagar',            area:'T Nagar',             stars:3, basePrice:1800,  rating:3.8, amenities:['WiFi','AC','Breakfast']},
    {name:'FabHotel Prime Pearl',                 area:'Egmore',              stars:3, basePrice:1900,  rating:3.7, amenities:['WiFi','AC','Restaurant']},
    {name:'Zostel Chennai (Hostel)',              area:'Triplicane',          stars:2, basePrice:700,   rating:4.3, amenities:['WiFi','Common Room','Backpacker']},
  ],
  hyderabad: [
    {name:'Taj Falaknuma Palace',                area:'Falaknuma',           stars:5, basePrice:32000, rating:4.9, amenities:['WiFi','Pool','Heritage Palace','Royal Suite']},
    {name:'ITC Kohenur',                          area:'HITEC City',          stars:5, basePrice:11500, rating:4.7, amenities:['WiFi','Pool','Spa','LEED Platinum']},
    {name:'Trident Hyderabad',                    area:'HITEC City',          stars:5, basePrice:9800,  rating:4.6, amenities:['WiFi','Pool','Spa','Gym']},
    {name:'Park Hyatt Hyderabad',                 area:'Banjara Hills',       stars:5, basePrice:11800, rating:4.7, amenities:['WiFi','Pool','Spa','Tian Restaurant']},
    {name:'Novotel HCC',                          area:'HITEC City',          stars:5, basePrice:6800,  rating:4.4, amenities:['WiFi','Pool','Convention Center']},
    {name:'Lemon Tree Premier HITEC City',        area:'HITEC City',          stars:4, basePrice:4900,  rating:4.3, amenities:['WiFi','Pool','Gym']},
    {name:'Treebo Trend Hometel',                 area:'Begumpet',            stars:3, basePrice:2300,  rating:4.0, amenities:['WiFi','AC','Breakfast']},
    {name:'OYO Townhouse Hi-Tech City',           area:'Madhapur',            stars:3, basePrice:1700,  rating:3.8, amenities:['WiFi','AC','Breakfast']},
    {name:'Zostel Hyderabad (Hostel)',            area:'Banjara Hills',       stars:2, basePrice:700,   rating:4.3, amenities:['WiFi','Common Room','Backpacker']},
  ],
  kolkata: [
    {name:'The Oberoi Grand, Kolkata',            area:'Jawaharlal Nehru Rd', stars:5, basePrice:13500, rating:4.7, amenities:['WiFi','Pool','Heritage','Spa']},
    {name:'ITC Royal Bengal',                      area:'New Town',            stars:5, basePrice:11800, rating:4.7, amenities:['WiFi','Pool','Spa','LEED Platinum']},
    {name:'Taj Bengal',                            area:'Alipore',             stars:5, basePrice:9800,  rating:4.6, amenities:['WiFi','Pool','Spa','Sonargaon']},
    {name:'JW Marriott Hotel Kolkata',            area:'Prafulla Kanan',      stars:5, basePrice:8200,  rating:4.5, amenities:['WiFi','Pool','Spa','Gym']},
    {name:'Hyatt Regency Kolkata',                area:'Salt Lake',           stars:5, basePrice:7200,  rating:4.4, amenities:['WiFi','Pool','Gym','Restaurant']},
    {name:'Lemon Tree Premier, Kolkata',          area:'Salt Lake',           stars:4, basePrice:4800,  rating:4.3, amenities:['WiFi','Pool','Gym']},
    {name:'Treebo Trend Park Plaza',              area:'Park Street',         stars:3, basePrice:2400,  rating:4.0, amenities:['WiFi','AC','Breakfast']},
    {name:'OYO Townhouse Park Street',            area:'Park Street',         stars:3, basePrice:1700,  rating:3.8, amenities:['WiFi','AC','Breakfast']},
    {name:'Zostel Kolkata (Hostel)',              area:'Park Street',         stars:2, basePrice:650,   rating:4.3, amenities:['WiFi','Common Room','Backpacker']},
  ],
  agra: [
    {name:'The Oberoi Amarvilas, Agra',           area:'Taj East Gate Road',  stars:5, basePrice:42000, rating:4.9, amenities:['WiFi','Pool','Spa','Taj View Rooms','Butler']},
    {name:'ITC Mughal, A Luxury Collection',      area:'Taj Ganj',            stars:5, basePrice:13800, rating:4.7, amenities:['WiFi','Pool','Kaya Kalp Spa','35-acre']},
    {name:'Taj Hotel & Convention Centre, Agra',  area:'Tajganj',             stars:5, basePrice:14200, rating:4.7, amenities:['WiFi','Pool','Spa','Taj View']},
    {name:'Trident Agra',                          area:'Fatehabad Road',      stars:5, basePrice:9500,  rating:4.5, amenities:['WiFi','Pool','Gym','Spa']},
    {name:'Radisson Hotel Agra',                   area:'Fatehabad Road',      stars:4, basePrice:5800,  rating:4.3, amenities:['WiFi','Pool','Gym','Restaurant']},
    {name:'Treebo Trend Crystal Inn',              area:'Fatehabad Road',      stars:3, basePrice:2200,  rating:4.0, amenities:['WiFi','AC','Breakfast','Taj View']},
    {name:'OYO Townhouse 070 Tajganj',             area:'Tajganj',             stars:3, basePrice:1700,  rating:3.8, amenities:['WiFi','AC','Breakfast']},
    {name:'Zostel Agra (Hostel)',                  area:'Tajganj',             stars:2, basePrice:600,   rating:4.4, amenities:['WiFi','Rooftop Taj View','Backpacker']},
  ],
  udaipur: [
    {name:'The Oberoi Udaivilas',                  area:'Lake Pichola',        stars:5, basePrice:48000, rating:4.9, amenities:['WiFi','Pool','Lake View','Heritage','Butler']},
    {name:'Taj Lake Palace, Udaipur',              area:'Lake Pichola Island', stars:5, basePrice:55000, rating:4.9, amenities:['WiFi','Heritage Palace','Lake Surround','Boat Access']},
    {name:'Taj Aravali Resort & Spa',              area:'Mavli Road',          stars:5, basePrice:14800, rating:4.7, amenities:['WiFi','Pool','Spa','Aravali Hills']},
    {name:'The Leela Palace Udaipur',              area:'Lake Pichola',        stars:5, basePrice:32000, rating:4.8, amenities:['WiFi','Pool','Lake View','Spa','Heritage']},
    {name:'Trident Udaipur',                        area:'Haridasji Ki Magri', stars:5, basePrice:9800,  rating:4.6, amenities:['WiFi','Pool','Gym','Lake View']},
    {name:'Radisson Blu Udaipur Palace Resort',    area:'Fatehsagar',          stars:5, basePrice:8500,  rating:4.4, amenities:['WiFi','Pool','Spa','Gym']},
    {name:'Treebo Trend Garden Hotel',             area:'Bhattiyani Chohatta', stars:3, basePrice:2200,  rating:4.0, amenities:['WiFi','AC','Breakfast']},
    {name:'Zostel Udaipur (Hostel)',                area:'Hanuman Ghat',        stars:2, basePrice:700,   rating:4.5, amenities:['WiFi','Lake View','Rooftop','Backpacker']},
  ],
  pondicherry: [
    {name:'The Promenade Pondicherry',              area:'Rock Beach',          stars:5, basePrice:9800,  rating:4.5, amenities:['WiFi','Pool','Sea View','French Quarter']},
    {name:'Le Pondy Beach Resort',                  area:'Kanagachettikulam',   stars:5, basePrice:8500,  rating:4.4, amenities:['WiFi','Beach','Pool','Spa']},
    {name:'Palais De Mahe',                          area:'White Town',          stars:4, basePrice:6500,  rating:4.5, amenities:['WiFi','Pool','Heritage','French Colonial']},
    {name:"Hotel de l'Orient",                       area:'White Town',          stars:4, basePrice:5800,  rating:4.4, amenities:['WiFi','Heritage','French Cuisine']},
    {name:'Lemon Tree Pondicherry',                  area:'Mission Street',      stars:4, basePrice:4500,  rating:4.2, amenities:['WiFi','Pool','Gym']},
    {name:'Treebo Trend Maison Radha',              area:'Goubert Avenue',      stars:3, basePrice:2400,  rating:4.0, amenities:['WiFi','AC','Breakfast']},
    {name:'OYO Townhouse Auroville Beach',          area:'Auroville',           stars:3, basePrice:1900,  rating:3.8, amenities:['WiFi','AC','Beach Access']},
    {name:'Micasa Backpackers (Hostel)',             area:'White Town',          stars:2, basePrice:600,   rating:4.4, amenities:['WiFi','Common Room','Backpacker']},
  ],
};

/* Real per-city surge multipliers for cab pricing */
export const CITY_CAB_MULTIPLIER = {
  delhi:1.15, mumbai:1.20, bangalore:1.15, chennai:1.05, kolkata:1.05, hyderabad:1.10,
  pune:1.10, ahmedabad:1.05, jaipur:1.00, goa:1.10, kochi:1.05, lucknow:0.95,
  agra:0.95, varanasi:0.90, udaipur:1.00, shimla:1.10, manali:1.15, ooty:1.10,
  rishikesh:1.00, darjeeling:1.10, leh:1.30, jodhpur:0.95, jaisalmer:1.05, hampi:0.95,
  munnar:1.10, mysore:1.00, coimbatore:0.95, vizag:1.00, shillong:1.05, gangtok:1.05,
  amritsar:0.95, kanyakumari:0.95, pondicherry:1.05, bhubaneswar:0.95,
};

/**
 * generateHotels(city, days, persona) — returns full hotel cards with bookingPlatforms,
 * SRM-specific options injected at the TOP when applicable.
 */
export function generateHotels(city, days, persona) {
  const cityKey = String(city || '').toLowerCase().trim().replace(/[^a-z\s]/g,'');

  let hotelList = [];
  for (const [k, list] of Object.entries(REAL_HOTELS_BY_CITY)) {
    if (cityKey.includes(k) || k.includes(cityKey)) { hotelList = [...list]; break; }
  }

  if (!hotelList.length) {
    hotelList = [
      {name:`Taj ${city}`,            area:'City Center',        stars:5, basePrice:9800,  rating:4.6, amenities:['WiFi','Pool','Spa','Restaurant','Gym']},
      {name:`ITC Hotel ${city}`,       area:'Business District',  stars:5, basePrice:8500,  rating:4.5, amenities:['WiFi','Pool','Spa','Restaurant']},
      {name:`Radisson Blu ${city}`,    area:'Central',            stars:5, basePrice:7200,  rating:4.4, amenities:['WiFi','Pool','Gym','Spa']},
      {name:`Novotel ${city}`,         area:'Central',            stars:4, basePrice:5800,  rating:4.3, amenities:['WiFi','Pool','Gym','Restaurant']},
      {name:`Lemon Tree Premier ${city}`,area:'City Center',      stars:4, basePrice:4500,  rating:4.2, amenities:['WiFi','Pool','Gym']},
      {name:`Holiday Inn ${city}`,      area:'Central',           stars:4, basePrice:4200,  rating:4.2, amenities:['WiFi','Pool','Gym']},
      {name:`Treebo Trend ${city} Inn`, area:'Central',           stars:3, basePrice:2200,  rating:4.0, amenities:['WiFi','AC','Breakfast']},
      {name:`FabHotel Prime ${city}`,   area:'Central',           stars:3, basePrice:1700,  rating:3.8, amenities:['WiFi','AC','Restaurant']},
      {name:`OYO Townhouse ${city}`,    area:'Central',           stars:3, basePrice:1500,  rating:3.7, amenities:['WiFi','AC','Breakfast']},
      {name:`Zostel ${city} (Hostel)`,  area:'Central',           stars:2, basePrice:650,   rating:4.3, amenities:['WiFi','Common Room','Breakfast','Backpacker']},
    ];
  }

  let filtered;
  if (persona === 'luxury') filtered = hotelList.filter(h => h.stars >= 4);
  else if (persona === 'adventure') filtered = hotelList.filter(h => h.basePrice <= 6000);
  else if (persona === 'family') filtered = hotelList.filter(h => h.stars >= 3 && h.stars <= 5);
  else filtered = hotelList;

  const seenNames = new Set();
  const dedupedFiltered = filtered.filter(h => {
    const k = (h.name||'').toLowerCase();
    if (seenNames.has(k)) return false;
    seenNames.add(k); return true;
  });

  const srmKey = isSRMCity(city);
  const srmList = srmKey ? (SRM_SPECIFIC_HOTELS[srmKey] || []) : [];

  const hotels = [...srmList, ...dedupedFiltered.map(h => ({
    name:h.name, stars:h.stars, basePrice:h.basePrice, rating:h.rating, amenities:h.amenities,
    address: h.area ? `${h.area}, ${city}` : city,
  }))];

  const checkinDate = new Date().toISOString().split('T')[0];
  const checkoutDate = new Date(Date.now() + (days||3)*86400000).toISOString().split('T')[0];
  const cityEnc = encodeURIComponent(city);
  const searchUrl = `https://www.booking.com/searchresults.html?ss=${cityEnc}&checkin=${checkinDate}&checkout=${checkoutDate}&group_adults=2&no_rooms=1`;

  return hotels.map((h, i) => {
    const variationPct = 0.9 + ((i * 7) % 20) / 100;
    const ppn = Math.round((h.basePrice || 0) * variationPct);
    const isHostel = !!h.applyRequired;
    const platforms = h.srmOfficial && h.officialUrl
      ? [{name: isHostel ? 'Apply on SRMIST Portal' : 'SRM Official', url: h.officialUrl, prefilled:true, srmOfficial:true}]
      : [];
    if (!isHostel) {
      platforms.push(
        {name:'Booking.com', url: searchUrl, prefilled:true},
        {name:'MakeMyTrip',  url: `https://www.makemytrip.com/hotels/hotel-listing?city=${cityEnc}&checkin=${checkinDate.replace(/-/g,'')}&checkout=${checkoutDate.replace(/-/g,'')}&roomStayQualifier=2e0e`, prefilled:true},
        {name:'Goibibo',     url: `https://www.goibibo.com/hotels/hotels-in-${String(city||'').toLowerCase().replace(/\s+/g,'-')}/?checkin=${checkinDate}&checkout=${checkoutDate}&adults_count=2&rooms_count=1`, prefilled:true},
        {name:'Agoda',       url: `https://www.agoda.com/search?city=${cityEnc}&checkIn=${checkinDate}&checkOut=${checkoutDate}&rooms=1&adults=2`, prefilled:true},
        {name:'Trivago',     url: `https://www.trivago.in/en-IN/srl?search=${cityEnc}&dr=${checkinDate}--${checkoutDate}&pa=2`, prefilled:true},
        {name:'OYO',         url: `https://www.oyorooms.com/search?location=${cityEnc}&checkin=${checkinDate}&checkout=${checkoutDate}`, prefilled:true},
      );
    }
    return {
      id: `HT${(h.name||'X').replace(/\s+/g,'').slice(0,8)}${i}`,
      name: h.name, stars: h.stars,
      price_per_night: ppn,
      total_price: ppn * (days || 3),
      rating: (typeof h.rating === 'number' ? h.rating : 4.0).toFixed(1),
      amenities: h.amenities,
      address: h.address || '',
      description: h.description || '',
      bookingUrl: isHostel ? (h.applyUrl || h.officialUrl || '#') : searchUrl,
      bookingPlatforms: platforms,
      image: h.image || '',
      currency: '₹',
      srmOfficial: !!h.srmOfficial,
      applyRequired: !!h.applyRequired,
      applyUrl: h.applyUrl || '',
      hostelType: h.hostelType || '',
    };
  }).sort((a, b) => {
    if (a.srmOfficial && !b.srmOfficial) return -1;
    if (!a.srmOfficial && b.srmOfficial) return 1;
    return a.price_per_night - b.price_per_night;
  });
}

/**
 * generateCabs(city) — full real provider list (Ola, Uber, Rapido, BluSmart, Meru, InDrive)
 */
export function generateCabs(city) {
  const cityKey = String(city||'').toLowerCase().trim().replace(/[^a-z\s]/g,'').trim();
  let mult = 1.0;
  for (const [k, m] of Object.entries(CITY_CAB_MULTIPLIER)) {
    if (cityKey.includes(k) || k.includes(cityKey)) { mult = m; break; }
  }
  const cityEnc = encodeURIComponent(city);

  // Realistic 2024-25 Indian ride-hail base fares & per-km rates
  // Sources: Ola/Uber app rate cards, Rapido helpdesk, BluSmart fixed pricing
  const providers = [
    {name:'Ola',types:[
      {type:'Auto',baseFare:35,perKm:13,minFare:60,rating:4.0},
      {type:'Mini',baseFare:90,perKm:14,minFare:130,rating:4.1},
      {type:'Prime Sedan',baseFare:110,perKm:17,minFare:170,rating:4.3},
      {type:'Prime SUV',baseFare:160,perKm:22,minFare:240,rating:4.4},
    ],url:`https://book.olacabs.com/?serviceType=p2p&utm_source=smartroute`,iconColor:'#bef264',about:"India's largest ride-hailing platform — covers 250+ cities"},
    {name:'Uber',types:[
      {type:'Uber Auto',baseFare:30,perKm:12,minFare:50,rating:4.1},
      {type:'UberGo',baseFare:85,perKm:14,minFare:125,rating:4.2},
      {type:'Premier',baseFare:115,perKm:18,minFare:175,rating:4.4},
      {type:'UberXL',baseFare:150,perKm:21,minFare:230,rating:4.4},
    ],url:`https://m.uber.com/ul/?action=setPickup&pickup=my_location&dropoff[formatted_address]=${cityEnc}`,iconColor:'#000',about:'Global ride-share — most reliable in metros and airports'},
    {name:'Rapido',types:[
      {type:'Bike Taxi',baseFare:20,perKm:6,minFare:40,rating:4.0},
      {type:'Rapido Auto',baseFare:30,perKm:10,minFare:55,rating:3.9},
      {type:'Rapido Cab Mini',baseFare:75,perKm:13,minFare:115,rating:4.0},
    ],url:`https://www.rapido.bike/`,iconColor:'#fbbf24',about:"India's #1 bike taxi — fastest in city traffic"},
    {name:'BluSmart',types:[
      {type:'BluSmart EV Sedan',baseFare:95,perKm:15,minFare:155,rating:4.6},
      {type:'BluSmart EV SUV',baseFare:135,perKm:19,minFare:215,rating:4.6},
    ],url:`https://blu-smart.com/`,iconColor:'#0ea5e9',about:'All-electric, no surge pricing, available in Delhi-NCR and Bengaluru'},
    {name:'Meru',types:[
      {type:'Meru Sedan',baseFare:105,perKm:16,minFare:165,rating:4.2},
      {type:'Meru SUV',baseFare:140,perKm:20,minFare:220,rating:4.3},
    ],url:`https://www.meru.in/`,iconColor:'#dc2626',about:"India's pioneer radio taxi — fixed metered fares"},
    {name:'InDrive',types:[
      {type:'InDrive Bid Cab',baseFare:65,perKm:12,minFare:95,rating:4.1},
    ],url:`https://indrive.com/`,iconColor:'#c5e600',about:'Set your own fare — bid-based pricing'},
  ];

  const results = [];
  for (const prov of providers) {
    for (const t of prov.types) {
      const adjBase = Math.round(t.baseFare * mult);
      const adjPerKm = Math.round(t.perKm * mult * 10) / 10;
      const eta10 = adjBase + Math.round(adjPerKm * 10);
      const eta20 = adjBase + Math.round(adjPerKm * 20);
      results.push({
        id: `CB${prov.name.slice(0,3).toUpperCase()}${results.length}`,
        provider: prov.name,
        provider_about: prov.about,
        type: t.type,
        price_per_km: adjPerKm,
        base_fare: adjBase,
        min_fare: Math.round(t.minFare * mult),
        rating: t.rating.toFixed(1),
        bookingUrl: prov.url,
        estimated_10km: eta10,
        estimated_20km: eta20,
        bookingPlatforms: [
          {name: `Open ${prov.name}`, url: prov.url, prefilled:true},
          {name:'Google Maps', url:`https://www.google.com/maps/dir/?api=1&destination=${cityEnc}&travelmode=driving`, prefilled:true},
        ],
      });
    }
  }
  return results.sort((a,b) => a.estimated_10km - b.estimated_10km);
}
