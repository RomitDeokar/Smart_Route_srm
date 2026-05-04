/* ════════════════════════════════════════════════════════════════
   _shared/transport.js — Flight & Train search (real airlines,
   IRCTC train rosters), merged from GitHub src/index.tsx.
   ════════════════════════════════════════════════════════════════ */

import { getDistance, getIATA } from './cities.js';

export function generateFlights(origin, dest, date) {
  const dist = getDistance(origin, dest);
  const oIATA = getIATA(origin);
  const dIATA = getIATA(dest);

  // Realistic Indian domestic airline pricing model based on 2024-2025 market data:
  //  - Floor cost ~₹2,200-2,800 (under-300 km hops still attract regulatory floor)
  //  - Linear ₹3.5-5.2/km depending on premium-ness, capped & with reasonable spread
  //  - Premium carriers (Vistara/AI) charge ~25% more, LCC (SpiceJet/IX) ~15% less
  const airlines = [
    {name:'IndiGo',           code:'6E', perKm:4.0, floor:2400, rating:4.1, fnRange:[2000,8999], fleet:'A320neo / A321'},
    {name:'Air India',        code:'AI', perKm:4.6, floor:2900, rating:4.0, fnRange:[440,899],   fleet:'A320 / B787'},
    {name:'Vistara',          code:'UK', perKm:5.0, floor:3200, rating:4.4, fnRange:[800,999],   fleet:'A320neo / B787-9'},
    {name:'SpiceJet',         code:'SG', perKm:3.6, floor:2200, rating:3.7, fnRange:[100,499],   fleet:'B737 MAX'},
    {name:'Air India Express',code:'IX', perKm:3.7, floor:2300, rating:3.8, fnRange:[1100,1899], fleet:'B737-800'},
    {name:'Akasa Air',        code:'QP', perKm:3.9, floor:2350, rating:4.2, fnRange:[1100,1499], fleet:'B737 MAX 8'},
  ];
  const dateParam = date || new Date().toISOString().split('T')[0];
  const oEnc = encodeURIComponent(origin || '');
  const dEnc = encodeURIComponent(dest || '');

  return airlines.map((airline, i) => {
    // Variance ±8% (deterministic seed from route)
    const seed = ((origin||'').length * 31 + (dest||'').length * 17 + i * 73);
    const variancePct = (((seed % 17) - 8) / 100); // ~ -0.08 .. +0.08
    let basePrice = Math.round(Math.max(airline.floor, dist * airline.perKm) * (1 + variancePct));
    // Realistic cap: never above ₹18,000 for domestic economy hops < 2500km
    if (dist < 2500) basePrice = Math.min(basePrice, 14500);

    const fnSpan = airline.fnRange[1] - airline.fnRange[0];
    const fnSeed = seed % fnSpan;
    const flightNo = `${airline.code} ${airline.fnRange[0] + fnSeed}`;

    const schedules = [
      {h:6, m:'15'},  {h:7, m:'45'},  {h:9, m:'10'},  {h:11, m:'30'},
      {h:14, m:'05'}, {h:17, m:'25'}, {h:19, m:'50'}, {h:21, m:'35'}
    ];
    const slot = schedules[i % schedules.length];
    const depH = slot.h, depMin = slot.m;

    const totalMin = Math.max(60, Math.round((dist / 750) * 60) + 25);
    const durH = Math.floor(totalMin / 60);
    const durM = totalMin % 60;
    const arrTotal = depH * 60 + parseInt(depMin) + totalMin;
    const arrH = Math.floor(arrTotal / 60) % 24;
    const arrM = arrTotal % 60;

    const isNonstop = dist < 1700;
    // All 6 cards show economy by default (real-world default search) so prices are comparable
    const cls = { type: 'Economy', multiplier: 1 };
    const price = Math.round(basePrice * cls.multiplier);

    return {
      id: `FL${airline.code}${i}`, airline: airline.name, flight_no: flightNo,
      origin_code: oIATA, dest_code: dIATA,
      aircraft: airline.fleet,
      departure: `${String(depH).padStart(2,'0')}:${depMin}`,
      arrival: `${String(arrH).padStart(2,'0')}:${String(arrM).padStart(2,'0')}`,
      duration: `${durH}h ${String(durM).padStart(2,'0')}m`, price, currency: '₹',
      class: cls.type,
      stops: isNonstop ? 0 : ((i % 3 === 0) ? 1 : 0),
      rating: airline.rating.toFixed(1),
      bookingPlatforms: [
        {name:'Google Flights', url: `https://www.google.com/travel/flights?q=flights+from+${oEnc}+to+${dEnc}+on+${dateParam}&curr=INR`, icon:'google', prefilled:true},
        {name:'MakeMyTrip', url: `https://www.makemytrip.com/flight/search?itinerary=${oEnc}-${dEnc}-${dateParam.replace(/-/g,'/')}&tripType=O&paxType=A-1_C-0_I-0&intl=false&cabinClass=E`, prefilled:true},
        {name:'Skyscanner', url: `https://www.skyscanner.co.in/transport/flights/${oEnc}/${dEnc}/${dateParam.replace(/-/g,'')}/?adultsv2=1&cabinclass=economy`, prefilled:true},
        {name:'ixigo', url: `https://www.ixigo.com/search/result/flight?from=${oEnc}&to=${dEnc}&date=${dateParam}&adults=1&class=e`, prefilled:true},
        {name:'Cleartrip', url: `https://www.cleartrip.com/flights/results?adults=1&class=Economy&depart_date=${dateParam}&from=${oEnc}&to=${dEnc}`, prefilled:true},
        {name:'EaseMyTrip', url: `https://flight.easemytrip.com/FlightList/Index?from=${oEnc}&to=${dEnc}&ddate=${dateParam}&isow=true&adult=1&sc=E`, prefilled:true},
      ]
    };
  }).sort((a,b) => a.price - b.price);
}

export const REAL_TRAINS = {
  'delhi|mumbai': [
    {no:'12952', name:'New Delhi - Mumbai Central Rajdhani', depart:'16:25', duration:'15h 50m', classes:['1A','2A','3A'], speed:90},
    {no:'12954', name:'August Kranti Rajdhani Express',     depart:'17:20', duration:'17h 05m', classes:['1A','2A','3A'], speed:85},
    {no:'22210', name:'NDLS - MMCT Duronto Express',         depart:'22:55', duration:'15h 35m', classes:['1A','2A','3A','SL'], speed:88},
    {no:'12138', name:'Punjab Mail',                          depart:'05:25', duration:'25h 15m', classes:['2A','3A','SL'], speed:55},
  ],
  'mumbai|delhi': [
    {no:'12951', name:'Mumbai Central - New Delhi Rajdhani', depart:'17:00', duration:'15h 32m', classes:['1A','2A','3A'], speed:90},
    {no:'12953', name:'August Kranti Rajdhani Express',      depart:'17:40', duration:'16h 35m', classes:['1A','2A','3A'], speed:85},
    {no:'22209', name:'MMCT - NDLS Duronto Express',          depart:'23:00', duration:'15h 50m', classes:['1A','2A','3A','SL'], speed:88},
  ],
  'delhi|chennai': [
    {no:'12434', name:'Hazrat Nizamuddin - Chennai Rajdhani',  depart:'15:50', duration:'28h 25m', classes:['1A','2A','3A'], speed:80},
    {no:'12622', name:'Tamil Nadu Express',                    depart:'22:30', duration:'33h 00m', classes:['2A','3A','SL'], speed:65},
    {no:'12616', name:'Grand Trunk Express',                   depart:'18:30', duration:'36h 25m', classes:['2A','3A','SL'], speed:60},
  ],
  'chennai|delhi': [
    {no:'12433', name:'Chennai - Hazrat Nizamuddin Rajdhani',  depart:'06:10', duration:'28h 25m', classes:['1A','2A','3A'], speed:80},
    {no:'12621', name:'Tamil Nadu Express',                    depart:'22:30', duration:'33h 25m', classes:['2A','3A','SL'], speed:65},
  ],
  'delhi|kolkata': [
    {no:'12302', name:'Howrah Rajdhani Express',               depart:'16:50', duration:'17h 05m', classes:['1A','2A','3A'], speed:85},
    {no:'12314', name:'Sealdah Rajdhani Express',              depart:'16:25', duration:'17h 35m', classes:['1A','2A','3A'], speed:84},
    {no:'12382', name:'Poorva Express',                         depart:'08:15', duration:'23h 20m', classes:['2A','3A','SL'], speed:65},
  ],
  'kolkata|delhi': [
    {no:'12301', name:'Howrah - New Delhi Rajdhani Express',   depart:'16:55', duration:'17h 20m', classes:['1A','2A','3A'], speed:85},
  ],
  'mumbai|chennai': [
    {no:'12163', name:'Chennai Express (Dadar - MAS)',          depart:'20:30', duration:'24h 50m', classes:['2A','3A','SL'], speed:60},
    {no:'22159', name:'CSMT - MAS Superfast Express',           depart:'00:15', duration:'24h 40m', classes:['2A','3A','SL'], speed:62},
  ],
  'chennai|mumbai': [
    {no:'12164', name:'Chennai - Dadar Express',                depart:'06:50', duration:'24h 30m', classes:['2A','3A','SL'], speed:60},
  ],
  'delhi|bangalore': [
    {no:'22692', name:'KSR Bengaluru Rajdhani',                 depart:'20:45', duration:'33h 50m', classes:['1A','2A','3A'], speed:75},
    {no:'12628', name:'Karnataka Express',                       depart:'21:15', duration:'37h 00m', classes:['2A','3A','SL'], speed:60},
  ],
  'bangalore|delhi': [
    {no:'22691', name:'KSR Bengaluru - Hazrat Nizamuddin Rajdhani', depart:'20:00', duration:'33h 30m', classes:['1A','2A','3A'], speed:75},
  ],
  'chennai|bangalore': [
    {no:'12007', name:'MAS - MYS Shatabdi Express',             depart:'06:00', duration:'05h 00m', classes:['CC','EC'], speed:85},
    {no:'12027', name:'MAS - SBC Shatabdi Express',             depart:'06:00', duration:'04h 50m', classes:['CC','EC'], speed:88},
  ],
  'bangalore|chennai': [
    {no:'12028', name:'SBC - MAS Shatabdi Express',             depart:'16:30', duration:'04h 50m', classes:['CC','EC'], speed:88},
    {no:'12658', name:'SBC - MAS Mail Express',                 depart:'22:40', duration:'06h 30m', classes:['2A','3A','SL'], speed:67},
  ],
  'mumbai|goa': [
    {no:'10103', name:'Mandovi Express',                        depart:'06:55', duration:'11h 50m', classes:['2A','3A','SL'], speed:50},
    {no:'12051', name:'Madgaon Janshatabdi',                    depart:'05:25', duration:'08h 25m', classes:['CC','2S'], speed:70},
  ],
  'goa|mumbai': [
    {no:'10104', name:'Mandovi Express',                        depart:'09:30', duration:'12h 00m', classes:['2A','3A','SL'], speed:50},
  ],
  'delhi|jaipur': [
    {no:'12015', name:'Ajmer Shatabdi Express',                 depart:'06:05', duration:'04h 35m', classes:['CC','EC'], speed:75},
    {no:'12958', name:'Ahmedabad Swarna Jayanti Rajdhani',      depart:'19:55', duration:'05h 00m', classes:['1A','2A','3A'], speed:75},
  ],
  'jaipur|delhi': [
    {no:'12016', name:'Ajmer - New Delhi Shatabdi',             depart:'17:55', duration:'04h 30m', classes:['CC','EC'], speed:75},
  ],
  'agra|delhi': [
    {no:'12001', name:'Bhopal Shatabdi (return)',                depart:'14:25', duration:'01h 55m', classes:['CC','EC'], speed:100},
    {no:'12050', name:'Gatimaan Express',                        depart:'17:50', duration:'01h 40m', classes:['CC','EC'], speed:120},
  ],
  'delhi|agra': [
    {no:'12002', name:'New Delhi - Bhopal Shatabdi',             depart:'06:00', duration:'01h 55m', classes:['CC','EC'], speed:100},
    {no:'12049', name:'Gatimaan Express',                         depart:'08:10', duration:'01h 40m', classes:['CC','EC'], speed:120},
  ],
};

export function generateTrains(origin, dest) {
  const dist = getDistance(origin, dest);
  const oKey = String(origin||'').toLowerCase().replace(/[^a-z]/g,'');
  const dKey = String(dest||'').toLowerCase().replace(/[^a-z]/g,'');
  const irctcUrl = `https://www.irctc.co.in/nget/train-search`;
  const confirmtktUrl = `https://www.confirmtkt.com/train-search?from=${encodeURIComponent(origin||'')}&to=${encodeURIComponent(dest||'')}`;
  const railYatriUrl = `https://www.railyatri.in/booking/search?from=${encodeURIComponent(origin||'')}&to=${encodeURIComponent(dest||'')}`;

  let realRoute = [];
  for (const [k, trains] of Object.entries(REAL_TRAINS)) {
    const [from, to] = k.split('|');
    if ((oKey.includes(from) || from.includes(oKey)) && (dKey.includes(to) || to.includes(dKey))) {
      realRoute = trains; break;
    }
  }

  // Realistic IRCTC fare ladder (₹/km) per coach class (2024-25 schedule).
  // 3A is the everyday default — keeps price headline reasonable.
  const classMultipliers = {'1A':3.0,'2A':1.7,'3A':1.1,'SL':0.45,'CC':0.9,'EC':1.6,'2S':0.35};
  const classFloors      = {'1A':1500,'2A':900,'3A':650,'SL':250,'CC':400,'EC':800,'2S':150};

  if (realRoute.length) {
    return realRoute.map((t) => {
      // Default to most common booked class: 3A if available, else CC, else first.
      const cls = t.classes.includes('3A') ? '3A' : t.classes.includes('CC') ? 'CC' : t.classes[0];
      const baseRate = t.name.includes('Rajdhani') ? 1.25
                     : t.name.includes('Vande Bharat') ? 1.4
                     : t.name.includes('Shatabdi') ? 1.15
                     : t.name.includes('Duronto') ? 1.1
                     : 1.0;
      const raw = dist * baseRate * (classMultipliers[cls] || 1) * 0.85; // 0.85 IRCTC discount factor
      const price = Math.max(classFloors[cls] || 250, Math.round(raw / 5) * 5); // round to nearest ₹5
      return {
        id: `TR${t.no}`, train_name: t.name, train_no: t.no,
        departure: t.depart, duration: t.duration, price, currency: '₹',
        class: cls, available_classes: t.classes,
        bookingUrl: irctcUrl,
        bookingPlatforms: [
          {name:'IRCTC',          url: irctcUrl,        prefilled:true},
          {name:'ConfirmTkt',     url: confirmtktUrl,   prefilled:true},
          {name:'RailYatri',      url: railYatriUrl,    prefilled:true},
          {name:'ixigo Trains',   url: `https://www.ixigo.com/search/result/train/${encodeURIComponent(origin||'')}/${encodeURIComponent(dest||'')}/`, prefilled:true},
          {name:'MakeMyTrip',     url:'https://www.makemytrip.com/railways/'},
          {name:'Cleartrip',      url:'https://www.cleartrip.com/trains'},
        ]
      };
    }).sort((a,b) => a.price - b.price);
  }

  const trainTypes = [
    {name:'Rajdhani Express',code:'RAJ',speedKmh:90,base:1.6,classes:['1A','2A','3A']},
    {name:'Shatabdi Express',code:'SHT',speedKmh:88,base:1.3,classes:['CC','EC']},
    {name:'Vande Bharat Express',code:'VBE',speedKmh:130,base:1.9,classes:['CC','EC']},
    {name:'Duronto Express',code:'DUR',speedKmh:85,base:1.4,classes:['1A','2A','3A','SL']},
    {name:'Garib Rath',code:'GR',speedKmh:75,base:0.7,classes:['3A','SL']},
    {name:'Superfast Express',code:'SF',speedKmh:70,base:0.9,classes:['2A','3A','SL']},
  ];
  const trainNoSeed = {RAJ:12259, SHT:12027, VBE:22439, DUR:12273, GR:12909, SF:12601};

  return trainTypes.filter(t => {
    if (dist < 300 && t.code === 'RAJ') return false;
    if (dist > 1500 && t.code === 'SHT') return false;
    return true;
  }).map((train, i) => {
    const totalMin = Math.max(120, Math.round((dist / train.speedKmh) * 60));
    const durH = Math.floor(totalMin / 60), durM = totalMin % 60;
    // Default 3A or CC for visible price (typical user pick) and apply IRCTC discount factor
    const cls = train.classes.includes('3A') ? '3A' : train.classes.includes('CC') ? 'CC' : train.classes[0];
    const raw = dist * train.base * (classMultipliers[cls] || 1) * 0.85;
    const price = Math.max(classFloors[cls] || 250, Math.round(raw / 5) * 5);
    const depH = [5,6,8,15,17,20][i % 6];
    const depMin = (i % 2 === 0) ? '00' : '30';
    const trainNo = (trainNoSeed[train.code] || 12000) + i;
    return {
      id: `TR${train.code}${i}`, train_name: `${train.name} (${origin}-${dest})`,
      train_no: String(trainNo),
      departure: `${String(depH).padStart(2,'0')}:${depMin}`,
      duration: `${durH}h ${String(durM).padStart(2,'0')}m`, price, currency: '₹',
      class: cls, available_classes: train.classes,
      bookingUrl: irctcUrl,
      bookingPlatforms: [
        {name:'IRCTC',          url: irctcUrl,        prefilled:true},
        {name:'ConfirmTkt',     url: confirmtktUrl,   prefilled:true},
        {name:'RailYatri',      url: railYatriUrl,    prefilled:true},
        {name:'ixigo Trains',   url: `https://www.ixigo.com/search/result/train/${encodeURIComponent(origin||'')}/${encodeURIComponent(dest||'')}/`, prefilled:true},
        {name:'MakeMyTrip',     url:'https://www.makemytrip.com/railways/'},
        {name:'Cleartrip',      url:'https://www.cleartrip.com/trains'},
      ]
    };
  }).sort((a,b) => a.price - b.price);
}
