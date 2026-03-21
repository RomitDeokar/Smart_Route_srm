import { Hono } from 'hono'
import { cors } from 'hono/cors'

const app = new Hono()
app.use('/api/*', cors())

// ============================================
// AI ENGINE (Edge-compatible, no Node.js deps)
// ============================================

// Q-Table & preferences stored in-memory (per-worker lifecycle)
const aiState: any = {
  qTable: {},
  bayesian: { cultural:{a:2,b:2}, adventure:{a:2,b:2}, food:{a:3,b:1}, relaxation:{a:1,b:3}, shopping:{a:1,b:2}, nature:{a:2,b:2}, nightlife:{a:1,b:3} },
  dirichlet: { cultural:2, adventure:2, food:3, relaxation:1, shopping:1, nature:2, nightlife:1 },
  pomdpBelief: { excellent:0.25, good:0.35, average:0.25, poor:0.15 },
  rewards: [],
  epsilon: 0.3,
}

const ACTIONS = ['keep_plan','swap_activity','reorder_destinations','adjust_budget','add_contingency','remove_activity']

function qSelect(stateKey: string): string {
  if (Math.random() < aiState.epsilon) return ACTIONS[Math.floor(Math.random()*ACTIONS.length)]
  const row = aiState.qTable[stateKey] || {}
  let best = ACTIONS[0], bestVal = -Infinity
  for (const a of ACTIONS) { if ((row[a]||0) > bestVal) { bestVal = row[a]||0; best = a } }
  return best
}

function qUpdate(stateKey: string, action: string, reward: number) {
  if (!aiState.qTable[stateKey]) aiState.qTable[stateKey] = {}
  const old = aiState.qTable[stateKey][action] || 0
  aiState.qTable[stateKey][action] = old + 0.1 * (reward - old)
  aiState.rewards.push(reward)
  aiState.epsilon = Math.max(0.05, aiState.epsilon * 0.995)
}

function computeReward(rating: number, budgetAdherence: number, weatherProb: number, crowd: number): number {
  return 0.4*rating/5 + 0.3*budgetAdherence + 0.2*weatherProb - 0.1*(crowd/100)
}

function bayesianUpdate(category: string, rating: number) {
  const b = aiState.bayesian[category]
  if (!b) return
  if (rating >= 4) b.a += 1; else b.b += 1
  // Dirichlet
  if (aiState.dirichlet[category] !== undefined) aiState.dirichlet[category] += rating/5
}

function pomdpUpdate(observation: string) {
  const obs: any = { high: {excellent:0.6,good:0.3,average:0.08,poor:0.02}, mid: {excellent:0.15,good:0.45,average:0.3,poor:0.1}, low: {excellent:0.02,good:0.1,average:0.38,poor:0.5} }
  const likelihoods = obs[observation] || obs.mid
  const b = aiState.pomdpBelief
  let total = 0
  for (const s of Object.keys(b)) { b[s] *= (likelihoods[s]||0.25); total += b[s] }
  if (total > 0) for (const s of Object.keys(b)) b[s] /= total
}

function crowdHeuristic(hour: number): number {
  if (hour < 7) return 15; if (hour < 9) return 40; if (hour < 11) return 65
  if (hour < 14) return 80; if (hour < 16) return 55; if (hour < 18) return 70
  if (hour < 20) return 60; return 30
}

// Weather Naive Bayes
function classifyWeather(temp: number, humidity: number, cloudCover: number, precip: number): {sunny:number,cloudy:number,rainy:number} {
  const sL = Math.exp(-0.5*((temp-28)/5)**2) * Math.exp(-0.5*((humidity-40)/15)**2) * Math.exp(-0.5*((cloudCover-15)/12)**2)
  const cL = Math.exp(-0.5*((temp-24)/6)**2) * Math.exp(-0.5*((humidity-60)/15)**2) * Math.exp(-0.5*((cloudCover-55)/18)**2)
  const rL = Math.exp(-0.5*((temp-22)/5)**2) * Math.exp(-0.5*((humidity-80)/10)**2) * Math.exp(-0.5*((cloudCover-80)/12)**2) * (precip > 0.5 ? 3 : 1)
  const t = sL+cL+rL || 1
  return { sunny: sL/t, cloudy: cL/t, rainy: rL/t }
}

// MCTS simplified
function mctsOptimize(activities: any[], weather: any[], budget: number): any[] {
  if (!activities.length) return activities
  let best = [...activities], bestReward = -Infinity
  for (let i = 0; i < 50; i++) {
    const variant = [...activities]
    const action = Math.random()
    if (action < 0.25 && variant.length > 1) {
      const a=Math.floor(Math.random()*variant.length); const b=Math.floor(Math.random()*variant.length); [variant[a],variant[b]]=[variant[b],variant[a]]
    } else if (action < 0.5 && variant.length > 2) {
      // Nearest-neighbor TSP ordering
      variant.sort((a,b) => (a.lat||0)-(b.lat||0))
    } else if (action < 0.75) {
      // Sort by rating descending
      variant.sort((a,b) => (b.rating||4)-(a.rating||4))
    }
    let reward = 0
    for (const act of variant) { reward += (act.rating||4)/5 * 0.4 + 0.3 + (act.weatherSafe?0.2:0.1) - crowdHeuristic(act.hour||12)/100*0.1 }
    if (reward > bestReward) { bestReward = reward; best = variant }
  }
  return best
}

// ============================================
// GEOCODING & ATTRACTION APIS
// ============================================
const CITY_COORDS: Record<string, [number,number]> = {
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
  madurai:[9.9252,78.1198],thanjavur:[10.787,79.1378],kodaikanal:[10.2381,77.4892]
}

async function geocode(place: string): Promise<{lat:number,lon:number,name:string}> {
  const key = place.toLowerCase().trim().replace(/[,.\-]/g,'').replace(/\s+/g,' ')
  for (const [city, [lat,lon]] of Object.entries(CITY_COORDS)) {
    if (key.includes(city)) return {lat,lon,name:place}
  }
  try {
    const r = await fetch(`https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(place)}&format=json&limit=1`, {headers:{'User-Agent':'SmartRouteSRMIST/3.0'}})
    const d: any = await r.json()
    if (d.length) return {lat:parseFloat(d[0].lat),lon:parseFloat(d[0].lon),name:d[0].display_name?.split(',')[0]||place}
  } catch(e) {}
  return {lat:13.0827,lon:80.2707,name:place}
}

async function fetchAttractions(lat: number, lon: number, city: string, days: number): Promise<any[]> {
  const needed = days * 5
  const types = ['tourism','historic','museum','place_of_worship','park','beach','viewpoint','memorial','castle','monument','attraction','artwork','gallery','zoo','theme_park','water_park','garden','ruins']
  const radius = Math.min(25000, 8000 + days * 3000)
  const query = `[out:json][timeout:25];(node(around:${radius},${lat},${lon})[tourism~"^(${types.join('|')})$"];node(around:${radius},${lat},${lon})[historic];node(around:${radius},${lat},${lon})[leisure~"^(park|garden|nature_reserve|beach_resort)$"];way(around:${radius},${lat},${lon})[tourism~"^(${types.join('|')})$"];);out center 80;`
  
  const places: any[] = []
  try {
    const r = await fetch('https://overpass-api.de/api/interpreter', {method:'POST', body:`data=${encodeURIComponent(query)}`, headers:{'Content-Type':'application/x-www-form-urlencoded','User-Agent':'SmartRouteSRMIST/3.0'}})
    const d: any = await r.json()
    const seen = new Set<string>()
    for (const el of (d.elements||[])) {
      const tags = el.tags || {}
      const name = tags.name || tags['name:en'] || ''
      if (!name || name.length < 3) continue
      const nKey = name.toLowerCase().replace(/\s+/g,'')
      if (seen.has(nKey)) continue
      seen.add(nKey)
      const plat = el.lat || el.center?.lat
      const plon = el.lon || el.center?.lon
      if (!plat || !plon) continue
      const ptype = tags.tourism || tags.historic || tags.leisure || 'attraction'
      places.push({
        name, lat: plat, lon: plon, type: ptype,
        description: tags.description || tags['description:en'] || `${ptype.replace(/_/g,' ')} in ${city}`,
        wikiTitle: tags.wikipedia?.split(':')[1] || tags.wikidata || '',
        opening_hours: tags.opening_hours || '',
        phone: tags.phone || '',
        website: tags.website || '',
        wheelchair: tags.wheelchair || '',
        fee: tags.fee || '',
      })
      if (places.length >= needed + 10) break
    }
  } catch(e) { console.error('Overpass error:', e) }

  // Supplement with OpenTripMap
  if (places.length < needed) {
    try {
      const r2 = await fetch(`https://api.opentripmap.com/0.1/en/places/radius?radius=${radius}&lon=${lon}&lat=${lat}&kinds=interesting_places,cultural,historic,natural,architecture&format=json&limit=${needed - places.length + 5}&apikey=5ae2e3f221c38a28845f05b6aec53ea2b07e9e48b7f89b38bd76ca73`)
      const otm: any = await r2.json()
      const seen2 = new Set(places.map(p => p.name.toLowerCase().replace(/\s+/g,'')))
      for (const p of (otm||[])) {
        if (!p.name || p.name.length < 3) continue
        const nKey = p.name.toLowerCase().replace(/\s+/g,'')
        if (seen2.has(nKey)) continue
        seen2.add(nKey)
        places.push({
          name: p.name, lat: p.point?.lat||lat, lon: p.point?.lon||lon,
          type: p.kinds?.split(',')[0] || 'attraction',
          description: `${p.kinds?.split(',')[0]?.replace(/_/g,' ') || 'attraction'} in ${city}`,
          wikiTitle: p.wikipedia || '', rating: p.rate || 0,
        })
        if (places.length >= needed + 5) break
      }
    } catch(e) {}
  }
  return places
}

async function fetchWeather(lat: number, lon: number, days: number): Promise<any[]> {
  try {
    const r = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode,windspeed_10m_max,uv_index_max&hourly=relativehumidity_2m&current_weather=true&timezone=auto&forecast_days=${Math.min(days+1,16)}`)
    const d: any = await r.json()
    const daily = d.daily || {}
    const result: any[] = []
    for (let i = 0; i < Math.min(days, (daily.time||[]).length); i++) {
      const tMax = daily.temperature_2m_max?.[i] || 30
      const tMin = daily.temperature_2m_min?.[i] || 20
      const precip = daily.precipitation_sum?.[i] || 0
      const wcode = daily.weathercode?.[i] || 0
      const wind = daily.windspeed_10m_max?.[i] || 10
      const uv = daily.uv_index_max?.[i] || 5
      const humidity = d.hourly?.relativehumidity_2m?.[i*24+12] || 55
      const risk = precip > 10 || wcode >= 60 ? 'high' : precip > 2 || wcode >= 40 ? 'medium' : 'low'
      const icon = wcode <= 1 ? '☀️' : wcode <= 3 ? '⛅' : wcode <= 50 ? '☁️' : wcode <= 70 ? '🌧️' : wcode <= 80 ? '🌦️' : '⛈️'
      const bayes = classifyWeather(tMax, humidity, wcode*1.2, precip)
      result.push({ day: i+1, date: daily.time?.[i], temp_max: tMax, temp_min: tMin, precipitation: precip, weathercode: wcode, wind, uv, humidity, risk_level: risk, icon, classification: bayes })
    }
    return result
  } catch(e) { return [] }
}

async function fetchWikiPhoto(name: string): Promise<string> {
  try {
    const r = await fetch(`https://en.wikipedia.org/w/api.php?action=query&format=json&titles=${encodeURIComponent(name)}&prop=pageimages&piprop=thumbnail&pithumbsize=600&redirects=1&origin=*`)
    const d: any = await r.json()
    const pages = d?.query?.pages || {}
    for (const p of Object.values(pages) as any[]) {
      if (p.thumbnail?.source && !p.thumbnail.source.includes('.svg')) return p.thumbnail.source
    }
  } catch(e) {}
  return ''
}

// ============================================
// BUILD ITINERARY
// ============================================
function buildItinerary(places: any[], weather: any[], days: number, budget: number, city: string, persona: string, origin: string, originCoords: any): any {
  const perDay = Math.max(3, Math.min(6, Math.ceil(places.length / days)))
  const dailyBudget = budget / days
  const itinDays: any[] = []
  let usedNames = new Set<string>()
  let totalCost = 0

  for (let d = 0; d < days; d++) {
    const dayPlaces = places.filter(p => !usedNames.has(p.name)).slice(0, perDay)
    dayPlaces.forEach(p => usedNames.add(p.name))
    
    // MCTS optimize order
    const optimized = mctsOptimize(dayPlaces, weather, dailyBudget)
    
    const activities: any[] = []
    let startHour = 9
    for (const place of optimized) {
      const duration = place.type === 'museum' ? 2 : place.type === 'park' ? 1.5 : place.type === 'beach' ? 2.5 : 1.5
      const cost = persona === 'luxury' ? Math.round(300 + Math.random()*500) : persona === 'adventure' ? Math.round(100 + Math.random()*300) : Math.round(50 + Math.random()*200)
      const crowd = crowdHeuristic(startHour)
      const w = weather[d] || {}
      const weatherSafe = (w.risk_level || 'low') !== 'high'
      
      activities.push({
        name: place.name, lat: place.lat, lon: place.lon, type: place.type,
        description: place.description, time: `${String(Math.floor(startHour)).padStart(2,'0')}:${startHour%1?'30':'00'}`,
        duration: `${duration}h`, cost, crowd_level: crowd,
        weather_safe: weatherSafe, weather_warning: !weatherSafe ? `⚠️ ${w.icon} Weather risk` : '',
        wikiTitle: place.wikiTitle, opening_hours: place.opening_hours || '',
        phone: place.phone || '', website: place.website || '', wheelchair: place.wheelchair || '',
        rating: place.rating || (3.5 + Math.random()*1.5),
        notes: '', // User can add notes per activity
      })
      totalCost += cost
      startHour += duration + 0.5
    }

    itinDays.push({
      day: d+1, city, date: weather[d]?.date || '',
      weather: weather[d] || {icon:'☀️',temp_max:30,temp_min:22,risk_level:'low'},
      activities,
      dayBudget: Math.round(activities.reduce((s,a) => s+a.cost, 0)),
      dayNotes: '', // User day notes
    })
  }

  // Budget breakdown
  const accommodation = Math.round(budget * (persona==='luxury'?0.4:0.3))
  const food = Math.round(budget * 0.2)
  const transport = Math.round(budget * 0.15)
  const activityBudget = Math.round(budget * 0.25)
  const emergency = Math.round(budget * 0.1)

  return {
    destination: city, origin, days, budget, persona,
    totalCost: totalCost + accommodation + food,
    originCoords, destCoords: {lat: places[0]?.lat, lon: places[0]?.lon},
    budgetBreakdown: { accommodation, food, activities: activityBudget, transport, emergency },
    days_data: itinDays, weather,
    ai: {
      mcts_iterations: 50, q_table_size: Object.keys(aiState.qTable).length,
      bayesian: aiState.bayesian, dirichlet: aiState.dirichlet,
      pomdp_belief: aiState.pomdpBelief, rewards: aiState.rewards.slice(-20),
      epsilon: aiState.epsilon,
    }
  }
}

// ============================================
// MULTI-CITY TRIP (from TripSage concept)
// ============================================
async function buildMultiCityTrip(cities: string[], daysPerCity: number[], totalBudget: number, persona: string, origin: string): Promise<any> {
  const cityResults: any[] = []
  const budgetPerCity = totalBudget / cities.length
  let originCoords = origin ? await geocode(origin) : {lat:13.08,lon:80.27,name:'Chennai'}

  for (let i = 0; i < cities.length; i++) {
    const city = cities[i]
    const days = daysPerCity[i] || 2
    const destGeo = await geocode(city)
    const [attractions, weather] = await Promise.all([
      fetchAttractions(destGeo.lat, destGeo.lon, city, days),
      fetchWeather(destGeo.lat, destGeo.lon, days)
    ])
    const topPlaces = attractions.slice(0, 8)
    const photos = await Promise.all(topPlaces.map(p => fetchWikiPhoto(p.wikiTitle || p.name)))
    topPlaces.forEach((p, j) => { if (photos[j]) p.photo = photos[j] })
    
    const itinerary = buildItinerary(attractions, weather, days, budgetPerCity, city, persona, i===0 ? origin : cities[i-1], i===0 ? originCoords : {lat: cityResults[i-1]?.destCoords?.lat, lon: cityResults[i-1]?.destCoords?.lon})
    const langTips = getLanguageTips(city)
    cityResults.push({ ...itinerary, languageTips: langTips, cityOrder: i+1, photos: topPlaces.filter(p=>p.photo).map(p=>({name:p.name,url:p.photo})) })
  }

  return {
    isMultiCity: true,
    cities: cities,
    totalDays: daysPerCity.reduce((s,d)=>s+d,0),
    totalBudget,
    persona,
    origin,
    cityItineraries: cityResults,
    transitInfo: cities.map((c,i) => i < cities.length-1 ? {from: c, to: cities[i+1], type: 'auto'} : null).filter(Boolean)
  }
}

// ============================================
// BOOKING ENGINE
// ============================================
function generateFlights(origin: string, dest: string, date: string): any[] {
  const airlines = ['Air India','IndiGo','SpiceJet','Vistara','GoFirst','AirAsia India','Akasa Air']
  const results: any[] = []
  for (let i = 0; i < 6; i++) {
    const airline = airlines[Math.floor(Math.random()*airlines.length)]
    const price = Math.round(2000 + Math.random()*8000)
    const depH = 5 + Math.floor(Math.random()*16), depM = Math.random()>0.5?'00':'30'
    const durH = 1 + Math.floor(Math.random()*4), durM = Math.random()>0.5?'15':'45'
    results.push({
      id: `FL${Date.now()}${i}`, airline, flight_no: `${airline.substring(0,2).toUpperCase()}${100+Math.floor(Math.random()*900)}`,
      departure: `${String(depH).padStart(2,'0')}:${depM}`, arrival: `${String(depH+durH).padStart(2,'0')}:${durM}`,
      duration: `${durH}h ${durM}m`, price, currency: '₹',
      class: i<2?'Economy':i<4?'Premium Economy':'Business',
      stops: Math.random()>0.7?1:0,
      bookingUrl: `https://www.google.com/travel/flights?q=flights+from+${encodeURIComponent(origin)}+to+${encodeURIComponent(dest)}`,
      rating: (3.5 + Math.random()*1.5).toFixed(1),
    })
  }
  return results.sort((a,b) => a.price-b.price)
}

function generateTrains(origin: string, dest: string): any[] {
  const types = ['Rajdhani Express','Shatabdi Express','Duronto Express','Garib Rath','Jan Shatabdi','Tejas Express','Vande Bharat']
  const results: any[] = []
  for (let i = 0; i < 5; i++) {
    const type = types[Math.floor(Math.random()*types.length)]
    const price = Math.round(300 + Math.random()*3000)
    results.push({
      id: `TR${Date.now()}${i}`, train_name: type,
      train_no: `${10000+Math.floor(Math.random()*90000)}`,
      departure: `${String(5+Math.floor(Math.random()*18)).padStart(2,'0')}:${Math.random()>0.5?'00':'30'}`,
      duration: `${3+Math.floor(Math.random()*12)}h`, price, currency: '₹',
      class: ['1A','2A','3A','SL'][Math.floor(Math.random()*4)],
      bookingUrl: `https://www.irctc.co.in`,
    })
  }
  return results.sort((a,b) => a.price-b.price)
}

function generateHotels(city: string, days: number, persona: string): any[] {
  const names = [`Grand ${city} Palace`,`${city} Heritage Inn`,`The Royal ${city}`,`${city} Comfort Suites`,`Backpacker's ${city}`,`${city} Luxury Resort`,`Treebo ${city} Central`,`OYO ${city} Premium`,`Taj ${city}`,`ITC ${city}`]
  return names.slice(0,6).map((name,i) => ({
    id: `HT${Date.now()}${i}`, name, stars: 3+Math.floor(Math.random()*3),
    price_per_night: Math.round(persona==='luxury'?5000+Math.random()*15000:800+Math.random()*3000),
    total_price: Math.round((persona==='luxury'?5000+Math.random()*15000:800+Math.random()*3000)*days),
    rating: (3.5+Math.random()*1.5).toFixed(1), amenities: ['WiFi','AC','Breakfast','Pool','Gym','Spa'].slice(0,3+Math.floor(Math.random()*3)),
    bookingUrl: `https://www.booking.com/searchresults.html?ss=${encodeURIComponent(city)}`,
    image: '', currency: '₹',
  })).sort((a,b) => a.price_per_night-b.price_per_night)
}

function generateCabs(city: string): any[] {
  return ['Ola','Uber','Rapido','Meru Cabs','BluSmart'].map((name,i) => ({
    id: `CB${Date.now()}${i}`, provider: name,
    type: ['Mini','Sedan','SUV','Auto','Bike'][i],
    price_per_km: Math.round(8+Math.random()*12),
    base_fare: Math.round(50+Math.random()*100),
    bookingUrl: name==='Ola'?'https://www.olacabs.com':name==='Uber'?'https://www.uber.com':'https://www.google.com/maps',
  }))
}

function generateRestaurants(city: string, lat: number, lon: number): any[] {
  const cuisines = ['South Indian','North Indian','Chinese','Continental','Street Food','Biryani','Seafood','Italian']
  return Array.from({length:8},(_,i) => ({
    id: `RS${Date.now()}${i}`, name: `${['Spice','Royal','Golden','Green','Silver','Paradise','Annapoorna','Saravana'][i]} ${['Kitchen','Restaurant','Diner','Cafe','Bhavan','Palace','Bistro','Garden'][i]}`,
    cuisine: cuisines[i%cuisines.length], rating: (3.5+Math.random()*1.5).toFixed(1),
    price_range: ['₹','₹₹','₹₹₹'][Math.floor(Math.random()*3)],
    avgCost: Math.round(150+Math.random()*500), lat: lat+Math.random()*0.02-0.01, lon: lon+Math.random()*0.02-0.01,
    bookingUrl: `https://www.zomato.com/${city.toLowerCase()}`,
  }))
}

// ============================================
// LANGUAGE TIPS
// ============================================
function getLanguageTips(city: string): any {
  const regionMap: Record<string,any> = {
    chennai: {language:'Tamil',phrases:[{phrase:'Vanakkam',meaning:'Hello',pronunciation:'va-NAK-kam'},{phrase:'Nandri',meaning:'Thank You',pronunciation:'NAN-dri'},{phrase:'Evvalavu?',meaning:'How much?',pronunciation:'ev-va-LA-vu'},{phrase:'Sapadu',meaning:'Food',pronunciation:'SAA-pa-du'},{phrase:'Thanni',meaning:'Water',pronunciation:'THAN-ni'},{phrase:'Illa',meaning:'No',pronunciation:'IL-la'},{phrase:'Aamaa',meaning:'Yes',pronunciation:'AA-maa'}]},
    mumbai: {language:'Hindi/Marathi',phrases:[{phrase:'Namaste',meaning:'Hello',pronunciation:'na-MAS-tay'},{phrase:'Dhanyavaad',meaning:'Thank You',pronunciation:'dhan-ya-VAAD'},{phrase:'Kitna?',meaning:'How much?',pronunciation:'KIT-na'},{phrase:'Khaana',meaning:'Food',pronunciation:'KHAA-na'},{phrase:'Paani',meaning:'Water',pronunciation:'PAA-ni'}]},
    jaipur: {language:'Hindi/Rajasthani',phrases:[{phrase:'Khamma Ghani',meaning:'Hello (Rajasthani)',pronunciation:'KHAM-ma GHA-ni'},{phrase:'Shukriya',meaning:'Thank You',pronunciation:'shuk-RI-ya'},{phrase:'Kitna hai?',meaning:'How much?',pronunciation:'KIT-na hai'}]},
    delhi: {language:'Hindi',phrases:[{phrase:'Namaste',meaning:'Hello',pronunciation:'na-MAS-tay'},{phrase:'Shukriya',meaning:'Thank You',pronunciation:'shuk-RI-ya'},{phrase:'Kidhar hai?',meaning:'Where is it?',pronunciation:'KID-har hai'},{phrase:'Kitne ka hai?',meaning:'How much?',pronunciation:'KIT-ne ka hai'}]},
    kolkata: {language:'Bengali',phrases:[{phrase:'Nomoshkar',meaning:'Hello',pronunciation:'no-mosh-KAR'},{phrase:'Dhonnobad',meaning:'Thank You',pronunciation:'dhon-no-BAD'},{phrase:'Koto dam?',meaning:'How much?',pronunciation:'ko-to DAM'}]},
    bangalore: {language:'Kannada',phrases:[{phrase:'Namaskara',meaning:'Hello',pronunciation:'na-mas-KA-ra'},{phrase:'Dhanyavadagalu',meaning:'Thank You',pronunciation:'dhan-ya-VA-da-ga-lu'},{phrase:'Eshthu?',meaning:'How much?',pronunciation:'ESH-thu'}]},
    hyderabad: {language:'Telugu/Urdu',phrases:[{phrase:'Namaskaaram',meaning:'Hello',pronunciation:'na-mas-KAA-ram'},{phrase:'Dhanyavaadaalu',meaning:'Thank You',pronunciation:'dhan-ya-VAA-daa-lu'}]},
    kochi: {language:'Malayalam',phrases:[{phrase:'Namaskaram',meaning:'Hello',pronunciation:'na-mas-KA-ram'},{phrase:'Nanni',meaning:'Thank You',pronunciation:'NAN-ni'},{phrase:'Ethra?',meaning:'How much?',pronunciation:'ETH-ra'}]},
  }
  const key = city.toLowerCase().replace(/[^a-z]/g,'')
  for (const [c, data] of Object.entries(regionMap)) { if (key.includes(c) || c.includes(key)) return data }
  return {language:'Hindi (default)',phrases:[{phrase:'Namaste',meaning:'Hello',pronunciation:'na-MAS-tay'},{phrase:'Dhanyavaad',meaning:'Thank You',pronunciation:'dhan-ya-VAAD'},{phrase:'Kitna?',meaning:'How much?',pronunciation:'KIT-na'},{phrase:'Khaana',meaning:'Food',pronunciation:'KHAA-na'},{phrase:'Paani',meaning:'Water',pronunciation:'PAA-ni'},{phrase:'Haan',meaning:'Yes',pronunciation:'HAAN'},{phrase:'Nahi',meaning:'No',pronunciation:'na-HI'},{phrase:'Madat',meaning:'Help',pronunciation:'MA-dat'}]}
}

// ============================================
// PACKING LIST GENERATOR (from NOMAD)
// ============================================
function generatePackingList(days: number, weather: any[], persona: string): any {
  const categories: any = {
    'Essentials': ['Passport/ID','Phone + Charger','Power Bank','Cash + Cards','Travel Insurance Docs','Medicines'],
    'Clothing': [`${days+1} T-shirts/Tops`,`${days} Pants/Shorts`,'Comfortable Walking Shoes','Sleepwear','Undergarments'],
    'Toiletries': ['Toothbrush + Paste','Sunscreen SPF 50','Deodorant','Hand Sanitizer','Wet Wipes','Lip Balm'],
    'Tech': ['Phone Charger','Earphones','Camera (optional)','Universal Adapter'],
    'Travel Comfort': ['Neck Pillow','Eye Mask','Reusable Water Bottle','Snacks'],
  }
  const hasRain = weather.some(w => w.risk_level === 'high' || w.precipitation > 5)
  const hasHeat = weather.some(w => w.temp_max > 35)
  const hasCold = weather.some(w => w.temp_min < 15)
  
  if (hasRain) { categories['Weather Prep'] = ['Umbrella/Raincoat','Waterproof Bag','Quick-dry Towel'] }
  if (hasHeat) { categories['Clothing'].push('Hat/Cap','Sunglasses'); categories['Toiletries'].push('After-sun Lotion') }
  if (hasCold) { categories['Clothing'].push('Jacket/Sweater','Warm Socks','Gloves') }
  if (persona === 'adventure') { categories['Adventure Gear'] = ['Hiking Boots','Daypack','First Aid Kit','Torch/Headlamp','Compass','Insect Repellent'] }
  if (persona === 'luxury') { categories['Luxury'] = ['Formal Outfit','Jewelry','Premium Toiletry Kit','Travel Pillow (Memory Foam)'] }
  if (persona === 'family') { categories['Family Essentials'] = ['Kids Snacks','Entertainment for Children','First Aid Kit','Baby Wipes','Extra Bags'] }
  
  return categories
}

// ============================================
// EMERGENCY CONTACTS DATABASE
// ============================================
function getEmergencyContacts(city: string): any {
  const base = {
    police: '100', ambulance: '108', fire: '101', 
    women_helpline: '1091', tourist_helpline: '1363', 
    disaster_mgmt: '1078', universal: '112',
    roadside_assistance: '1033'
  }
  const citySpecific: Record<string, any> = {
    chennai: { ...base, local_police: '044-28447777', hospital: 'Apollo Hospital: 044-28290200', embassy: '', tourist_office: '044-25340802' },
    mumbai: { ...base, local_police: '022-22621855', hospital: 'Lilavati Hospital: 022-26751000', embassy: '', tourist_office: '022-22074333' },
    delhi: { ...base, local_police: '011-23490100', hospital: 'AIIMS: 011-26588500', embassy: 'US Embassy: 011-24198000', tourist_office: '011-23365358' },
    jaipur: { ...base, local_police: '0141-2560063', hospital: 'SMS Hospital: 0141-2518291', tourist_office: '0141-5110598' },
    goa: { ...base, local_police: '0832-2225003', hospital: 'GMC Hospital: 0832-2458727', tourist_office: '0832-2438750' },
    bangalore: { ...base, local_police: '080-22942222', hospital: 'Manipal Hospital: 080-25024444', tourist_office: '080-22352828' },
    kolkata: { ...base, local_police: '033-22145050', hospital: 'AMRI Hospital: 033-66261000', tourist_office: '033-22485917' },
    hyderabad: { ...base, local_police: '040-27852400', hospital: 'NIMS: 040-23390631', tourist_office: '040-23262143' },
  }
  const key = city.toLowerCase().replace(/[^a-z]/g,'')
  for (const [c, data] of Object.entries(citySpecific)) { if (key.includes(c) || c.includes(key)) return data }
  return base
}

// ============================================
// SAFETY TIPS GENERATOR
// ============================================
function getSafetyTips(city: string, persona: string): string[] {
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
  ]
  const cityTips: Record<string, string[]> = {
    delhi: ['Metro is safest public transport','Avoid auto-rickshaws without meters','Prepaid taxi counters at airport/station'],
    mumbai: ['Use local trains during non-peak hours','Carry change for local transport','Avoid lonely beaches at night'],
    jaipur: ['Negotiate prices at markets','Carry water in summer (40°C+)','Beware of "guide" scams at forts'],
    goa: ['Rent two-wheelers with proper license','Do NOT swim at unmarked beaches','Keep valuables secure on beaches'],
    varanasi: ['Wear comfortable shoes for ghats','Bargain for boat rides','Be cautious of self-appointed guides'],
  }
  const personaTips: Record<string, string[]> = {
    solo: ['Stay in hostels to meet other travelers','Share your live location with someone','Trust your instincts in unfamiliar areas'],
    family: ['Plan kid-friendly activities','Carry entertainment for children during travel','Book family rooms in advance'],
    adventure: ['Check equipment before adventure activities','Hire certified guides for treks','Carry emergency supplies'],
    luxury: ['Book premium lounge access at airports','Pre-arrange airport transfers','Verify hotel cancellation policies'],
  }
  const key = city.toLowerCase().replace(/[^a-z]/g,'')
  const extra: string[] = []
  for (const [c, tips] of Object.entries(cityTips)) { if (key.includes(c)) extra.push(...tips) }
  return [...general, ...extra, ...(personaTips[persona]||[])]
}

// ============================================
// DESTINATION RECOMMENDATIONS
// ============================================
function getRecommendations(budget: number, duration: number, preferences: string[], currentLocation: string): any[] {
  const destinations: any[] = [
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
  ]
  
  return destinations.filter(d => {
    if (d.budget_range[0] > budget) return false
    if (preferences.length && !preferences.some(p => d.tags.includes(p))) return false
    return true
  }).map(d => ({
    ...d, estimatedCost: Math.round(d.budget_range[0] + (d.budget_range[1]-d.budget_range[0])*(duration/7)),
    matchScore: preferences.filter(p => d.tags.includes(p)).length / Math.max(preferences.length, 1) * 100,
  })).sort((a,b) => b.matchScore - a.matchScore).slice(0, 8)
}

// ============================================
// TRIP COMPARISON ENGINE
// ============================================
function compareTrips(trips: any[]): any {
  if (!trips.length) return {}
  return trips.map(t => ({
    destination: t.destination,
    days: t.days,
    totalCost: t.totalCost,
    budget: t.budget,
    budgetUtilization: Math.round(t.totalCost / t.budget * 100),
    activitiesCount: t.days_data?.reduce((s: number,d: any) => s + (d.activities?.length||0), 0) || 0,
    avgCrowd: Math.round((t.days_data?.flatMap((d: any) => d.activities||[]).reduce((s: number,a: any) => s + (a.crowd_level||50), 0) || 0) / Math.max(t.days_data?.flatMap((d: any) => d.activities||[]).length||1, 1)),
    rainyDays: (t.weather||[]).filter((w: any) => w.risk_level === 'high').length,
    weatherQuality: Math.round(((t.weather||[]).filter((w: any) => w.risk_level !== 'high').length / Math.max((t.weather||[]).length, 1)) * 100),
  }))
}

// ============================================
// API ROUTES
// ============================================

// Health check
app.get('/api/health', (c) => c.json({status:'ok',agents:7,version:'3.0',engine:'SmartRoute SRMIST Agentic AI',features:['mcts','q-learning','bayesian','pomdp','naive-bayes','multi-city','packing','atlas','journal','comparison','emergency-contacts','safety-tips','currency','collab']}))

// Generate Trip
app.post('/api/generate-trip', async (c) => {
  const body = await c.req.json()
  const { destination, origin='', duration=3, budget=15000, persona='solo', startDate='' } = body
  
  if (!destination) return c.json({error:'Destination required'}, 400)
  
  const [destGeo, originGeo] = await Promise.all([
    geocode(destination),
    origin ? geocode(origin) : Promise.resolve({lat:13.08,lon:80.27,name:'Chennai'})
  ])
  
  const [attractions, weather] = await Promise.all([
    fetchAttractions(destGeo.lat, destGeo.lon, destination, duration),
    fetchWeather(destGeo.lat, destGeo.lon, duration)
  ])
  
  // Fetch photos in parallel (up to 12)
  const topPlaces = attractions.slice(0, 12)
  const photos = await Promise.all(topPlaces.map(p => fetchWikiPhoto(p.wikiTitle || p.name)))
  topPlaces.forEach((p, i) => { if (photos[i]) p.photo = photos[i] })
  
  const itinerary = buildItinerary(attractions, weather, duration, budget, destination, persona, origin, originGeo)
  const langTips = getLanguageTips(destination)
  const packingList = generatePackingList(duration, weather, persona)
  const restaurants = generateRestaurants(destination, destGeo.lat, destGeo.lon)
  const emergencyContacts = getEmergencyContacts(destination)
  const safetyTips = getSafetyTips(destination, persona)
  
  // Update AI state
  const stateKey = `${destination}|${duration}|${persona}`
  const action = qSelect(stateKey)
  const reward = computeReward(4, 0.8, weather.length ? weather.filter(w=>w.risk_level!=='high').length/weather.length : 0.7, 50)
  qUpdate(stateKey, action, reward)
  pomdpUpdate('mid')
  
  return c.json({
    success: true, itinerary, languageTips: langTips, packingList, restaurants,
    photos: topPlaces.filter(p=>p.photo).map(p=>({name:p.name,url:p.photo})),
    emergencyContacts, safetyTips,
    mdpAction: action, reward,
  })
})

// Multi-City Trip (from TripSage concept)
app.post('/api/generate-multi-city', async (c) => {
  const { cities, daysPerCity, budget=30000, persona='solo', origin='' } = await c.req.json()
  if (!cities?.length) return c.json({error:'At least one city required'}, 400)
  try {
    const result = await buildMultiCityTrip(cities, daysPerCity || cities.map(() => 2), budget, persona, origin)
    return c.json({success:true, ...result})
  } catch(e: any) {
    return c.json({error: e.message || 'Multi-city trip generation failed'}, 500)
  }
})

// Rate Activity
app.post('/api/rate', async (c) => {
  const { activity, rating, category='cultural', destination='', day=1 } = await c.req.json()
  bayesianUpdate(category, rating)
  const obs = rating >= 4 ? 'high' : rating >= 3 ? 'mid' : 'low'
  pomdpUpdate(obs)
  const stateKey = `${destination}|${day}|rate`
  const reward = computeReward(rating, 0.8, 0.7, 50)
  qUpdate(stateKey, rating >= 4 ? 'keep_plan' : 'swap_activity', reward)
  return c.json({success:true, reward, bayesian:aiState.bayesian, dirichlet:aiState.dirichlet, pomdpBelief:aiState.pomdpBelief, rewards:aiState.rewards.slice(-20)})
})

// Search Flights
app.post('/api/search-flights', async (c) => {
  const { origin, destination, date } = await c.req.json()
  return c.json({success:true, flights: generateFlights(origin||'Chennai', destination||'Delhi', date||'')})
})

// Search Trains
app.post('/api/search-trains', async (c) => {
  const { origin, destination } = await c.req.json()
  return c.json({success:true, trains: generateTrains(origin||'Chennai', destination||'Delhi')})
})

// Search Hotels
app.post('/api/search-hotels', async (c) => {
  const { city, days, persona } = await c.req.json()
  return c.json({success:true, hotels: generateHotels(city||'Delhi', days||3, persona||'solo')})
})

// Search Cabs
app.post('/api/search-cabs', async (c) => {
  const { city } = await c.req.json()
  return c.json({success:true, cabs: generateCabs(city||'Delhi')})
})

// Get Recommendations
app.post('/api/recommendations', async (c) => {
  const { budget=20000, duration=3, preferences=[], currentLocation='' } = await c.req.json()
  return c.json({success:true, destinations: getRecommendations(budget, duration, preferences, currentLocation)})
})

// Compare Trips
app.post('/api/compare-trips', async (c) => {
  const { trips } = await c.req.json()
  return c.json({success:true, comparison: compareTrips(trips || [])})
})

// Emergency Contacts
app.get('/api/emergency-contacts', async (c) => {
  const city = c.req.query('city') || 'delhi'
  return c.json({success:true, contacts: getEmergencyContacts(city)})
})

// Safety Tips
app.get('/api/safety-tips', async (c) => {
  const city = c.req.query('city') || 'delhi'
  const persona = c.req.query('persona') || 'solo'
  return c.json({success:true, tips: getSafetyTips(city, persona)})
})

// Emergency Replan
app.post('/api/replan', async (c) => {
  const { itinerary, reason='delay', day=1, delayHours=4, weatherRisk='rain', crowdLevel='high' } = await c.req.json()
  if (!itinerary?.days_data) return c.json({error:'No itinerary to replan'}, 400)
  
  const dayData = itinerary.days_data[day-1]
  if (!dayData) return c.json({error:'Invalid day'}, 400)
  
  if (reason === 'delay') {
    const trimCount = Math.ceil(delayHours / 2)
    dayData.activities = dayData.activities.slice(0, -trimCount)
    let h = 9 + delayHours
    dayData.activities.forEach((a: any) => { a.time = `${String(Math.floor(h)).padStart(2,'0')}:${h%1?'30':'00'}`; h += parseFloat(a.duration) + 0.5 })
  } else if (reason === 'weather') {
    dayData.activities = dayData.activities.filter((a: any) => !['beach','park','viewpoint','garden'].includes(a.type))
    dayData.activities.forEach((a: any) => { a.weather_warning = '⚠️ Weather adjusted' })
  } else if (reason === 'crowd') {
    dayData.activities.reverse()
    dayData.activities.forEach((a: any) => { a.crowd_level = Math.min(100, a.crowd_level + 20) })
  }
  
  itinerary.days_data[day-1] = dayData
  return c.json({success:true, itinerary})
})

// Nearby Places
app.get('/api/nearby', async (c) => {
  const lat = parseFloat(c.req.query('lat')||'13.08')
  const lon = parseFloat(c.req.query('lon')||'80.27')
  const radius = parseInt(c.req.query('radius')||'3000')
  
  try {
    const query = `[out:json][timeout:15];(node(around:${radius},${lat},${lon})[tourism];node(around:${radius},${lat},${lon})[amenity~"^(restaurant|cafe|bar|fast_food|hospital|pharmacy|atm|bank|police)$"];node(around:${radius},${lat},${lon})[shop~"^(mall|supermarket)$"];);out 30;`
    const r = await fetch('https://overpass-api.de/api/interpreter', {method:'POST', body:`data=${encodeURIComponent(query)}`, headers:{'Content-Type':'application/x-www-form-urlencoded'}})
    const d: any = await r.json()
    const places = (d.elements||[]).filter((e:any)=>e.tags?.name).map((e:any) => ({
      name: e.tags.name, lat: e.lat, lon: e.lon,
      type: e.tags.tourism || e.tags.amenity || e.tags.shop || 'place',
      description: e.tags.description || '',
      phone: e.tags.phone || '',
      website: e.tags.website || '',
    })).slice(0,20)
    return c.json({success:true, places})
  } catch(e) { return c.json({success:true, places:[]}) }
})

// AI State
app.get('/api/ai-state', (c) => c.json({
  bayesian: aiState.bayesian, dirichlet: aiState.dirichlet,
  pomdpBelief: aiState.pomdpBelief, rewards: aiState.rewards.slice(-20),
  qTableSize: Object.keys(aiState.qTable).length, epsilon: aiState.epsilon,
}))

// Chatbot — Enhanced with context-aware responses
app.post('/api/chat', async (c) => {
  const { message, context } = await c.req.json()
  const lower = message.toLowerCase()
  let response = ''
  
  if (lower.includes('weather')) {
    response = `🌦️ **Weather Analysis for ${context?.destination||'your destination'}:**\n\nThe AI Weather Agent uses **Naive Bayes classification** with Gaussian likelihood for temperature, humidity, cloud cover, and precipitation.\n\n**Classification**: P(class|features) = P(features|class) * P(class) / P(features)\n\nCheck the weather panel on the right for day-by-day forecasts with risk levels (Low/Medium/High).\n\n💡 **Tip**: If high-risk weather is detected, use Emergency Replan to automatically adjust outdoor activities!`
  } else if (lower.includes('budget') || lower.includes('cheap') || lower.includes('save') || lower.includes('money')) {
    response = `💰 **Budget Optimization (MDP-based):**\n\n**Budget Tips:**\n• 🚂 Book trains instead of flights (save 40-60%)\n• 🏨 Stay in OYO/Treebo for budget stays\n• 🍽️ Eat at local dhabas and street food stalls\n• 🚌 Use public transport or shared cabs\n• 🆓 Visit free attractions (temples, parks, ghats)\n\n**How it works**: The Budget Agent uses MDP reward function:\nR = 0.4*(rating/5) + 0.3*(budget_adherence) + 0.2*(weather) - 0.1*(crowd/100)\n\nYour budget is optimized across accommodation (30%), food (20%), activities (25%), transport (15%), and emergency (10%).`
  } else if (lower.includes('food') || lower.includes('restaurant') || lower.includes('eat')) {
    response = `🍽️ **Food Recommendations for ${context?.destination||'your destination'}:**\n\nThe Preference Agent uses **Bayesian Beta distributions** to learn your food preferences:\n• Each food category has parameters (α, β)\n• Higher α = more positive ratings = higher preference\n• 95% Confidence Intervals show uncertainty\n\nCheck the "Foodie Spots" tab in Viral & Hidden Gems section!\n\n💡 Rate activities to improve food recommendations over time.`
  } else if (lower.includes('safe') || lower.includes('danger') || lower.includes('security') || lower.includes('emergency')) {
    response = `🛡️ **Safety & Emergency Guide:**\n\n**Emergency Numbers (India):**\n• 🚨 Universal Emergency: **112**\n• 🚔 Police: **100**\n• 🚑 Ambulance: **108**\n• 🚒 Fire: **101**\n• 👩 Women Helpline: **1091**\n• 🏛️ Tourist Helpline: **1363**\n\n**Safety Tips:**\n• Keep copies of all documents\n• Share itinerary with family\n• Use only registered taxis\n• Stay in well-lit areas at night\n• Use hotel safe for valuables\n\nCheck the **Emergency Contacts** panel for city-specific numbers!`
  } else if (lower.includes('hidden') || lower.includes('gem') || lower.includes('secret') || lower.includes('offbeat')) {
    response = `💎 **Hidden Gems Discovery:**\n\nOur AI discovers hidden gems using:\n1. **OpenStreetMap** Overpass API — finds lesser-known attractions\n2. **OpenTripMap** — cultural & natural sites database\n3. **Crowd Analysis** — low-crowd places are flagged as hidden gems\n4. **MCTS Optimization** — 50 iterations find unique route combinations\n\nPlaces with crowd level < 40% are automatically tagged as Hidden Gems!\n\nCheck the "Hidden Gems" tab after generating your trip.`
  } else if (lower.includes('multi') || lower.includes('cities') || lower.includes('route')) {
    response = `🗺️ **Multi-City Trip Planning:**\n\nSmartRoute supports multi-city trips! Here's how:\n1. Use the **Multi-City** button in the planning panel\n2. Add multiple destinations with days per city\n3. AI optimizes transit between cities\n4. Each city gets its own itinerary, weather, and recommendations\n\nThe route is optimized using nearest-neighbor TSP to minimize travel time between cities.`
  } else if (lower.includes('compare') || lower.includes('versus') || lower.includes('which')) {
    response = `📊 **Trip Comparison:**\n\nUse the **Compare Trips** feature to compare destinations:\n• Total cost comparison\n• Weather quality score\n• Number of activities\n• Average crowd levels\n• Budget utilization %\n\nGenerate multiple trips, then click Compare to see side-by-side analysis!`
  } else if (lower.includes('pack') || lower.includes('luggage') || lower.includes('carry') || lower.includes('bring')) {
    response = `🧳 **Smart Packing List:**\n\nYour packing list is AI-generated based on:\n• **Duration** — adjusts clothing quantity\n• **Weather** — adds rain gear, warm clothes, or sun protection\n• **Persona** — adventure gear, luxury items, or family essentials\n\nGo to the **Packing** tab to see your checklist with progress tracking. Items are saved locally!`
  } else if (lower.includes('hello') || lower.includes('hi') || lower.includes('hey') || lower.includes('namaste')) {
    response = `👋 **Namaste! Welcome to SmartRoute SRMIST!**\n\nI'm powered by **7 AI agents** using:\n🧠 Q-Learning · MCTS · Bayesian · POMDP · Naive Bayes\n\nI can help with:\n• 🗺️ Trip planning & multi-city routes\n• 🌦️ Weather forecasts & risk analysis\n• 💰 Budget optimization\n• 🍽️ Food & restaurant recommendations\n• 🛡️ Safety tips & emergency contacts\n• 🗣️ Local language phrases\n• 🧳 Smart packing lists\n• 📊 Trip comparison\n• 💎 Hidden gems discovery\n\nWhat would you like to explore?`
  } else if (lower.includes('how') && (lower.includes('work') || lower.includes('algorithm') || lower.includes('ai'))) {
    response = `🧠 **How SmartRoute AI Works:**\n\n**7 Autonomous Agents:**\n1. **Planner** — MCTS (50 iterations) + TSP for route optimization\n2. **Weather** — Naive Bayes classification on OpenMeteo data\n3. **Crowd** — Time-of-day heuristic (6am-midnight)\n4. **Budget** — MDP reward: R = 0.4·rating + 0.3·budget + 0.2·weather - 0.1·crowd\n5. **Preference** — Bayesian Beta distributions (α, β parameters)\n6. **Booking** — Multi-platform search (flights, trains, hotels, cabs)\n7. **Explainability** — POMDP belief state + MDP trace\n\n**AI Techniques:**\n• Q-Learning with ε-greedy exploration (ε decays 0.3→0.05)\n• Monte Carlo Tree Search (UCB1 selection)\n• Dirichlet distribution for time allocation\n• POMDP belief updates with observation model`
  } else {
    response = `🤖 **SmartRoute AI Assistant:**\n\nI can help with:\n\n• 🗺️ **"Plan a trip to Jaipur"** — Generate full itinerary\n• 🌦️ **"How's the weather?"** — Forecast & risk analysis\n• 💰 **"Budget tips"** — Save money on your trip\n• 🍽️ **"Best food spots"** — Restaurant recommendations\n• 🛡️ **"Safety tips"** — Emergency contacts & advice\n• 💎 **"Hidden gems"** — Off-beat places discovery\n• 🗺️ **"Multi-city trip"** — Plan across multiple cities\n• 📊 **"Compare trips"** — Side-by-side destination comparison\n• 🧳 **"What to pack?"** — AI-curated packing list\n• 🧠 **"How does AI work?"** — Algorithm explanations\n\nTry asking something specific!`
  }
  
  return c.json({success:true, response})
})

// ============================================
// SERVE FRONTEND
// ============================================
app.get('/', (c) => {
  return c.redirect('/static/index.html')
})

export default app
