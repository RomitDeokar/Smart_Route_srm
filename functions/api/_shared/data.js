/* ════════════════════════════════════════════════════════════════
   _shared/data.js — Static datasets used by Cloudflare Pages Funcs
   ════════════════════════════════════════════════════════════════ */

export const KNOWN_COORDS = {
  shillong:    { name: "Shillong",    country: "India", latitude: 25.5788, longitude: 91.8933 },
  goa:         { name: "Goa",         country: "India", latitude: 15.2993, longitude: 74.124  },
  ooty:        { name: "Ooty",        country: "India", latitude: 11.4102, longitude: 76.695  },
  munnar:      { name: "Munnar",      country: "India", latitude: 10.0889, longitude: 77.0595 },
  rishikesh:   { name: "Rishikesh",   country: "India", latitude: 30.0869, longitude: 78.2676 },
  udaipur:     { name: "Udaipur",     country: "India", latitude: 24.5854, longitude: 73.7125 },
  jaipur:      { name: "Jaipur",      country: "India", latitude: 26.9124, longitude: 75.7873 },
  delhi:       { name: "Delhi",       country: "India", latitude: 28.6139, longitude: 77.209  },
  mumbai:      { name: "Mumbai",      country: "India", latitude: 19.076,  longitude: 72.8777 },
  chennai:     { name: "Chennai",     country: "India", latitude: 13.0827, longitude: 80.2707 },
  bangalore:   { name: "Bangalore",   country: "India", latitude: 12.9716, longitude: 77.5946 },
  kolkata:     { name: "Kolkata",     country: "India", latitude: 22.5726, longitude: 88.3639 },
  hyderabad:   { name: "Hyderabad",   country: "India", latitude: 17.385,  longitude: 78.4867 },
  varanasi:    { name: "Varanasi",    country: "India", latitude: 25.3176, longitude: 83.0064 },
  manali:      { name: "Manali",      country: "India", latitude: 32.2396, longitude: 77.1887 },
  shimla:      { name: "Shimla",      country: "India", latitude: 31.1048, longitude: 77.1734 },
  darjeeling:  { name: "Darjeeling",  country: "India", latitude: 27.041,  longitude: 88.2663 },
  agra:        { name: "Agra",        country: "India", latitude: 27.1767, longitude: 78.0081 },
  kochi:       { name: "Kochi",       country: "India", latitude: 9.9312,  longitude: 76.2673 },
  pondicherry: { name: "Pondicherry", country: "India", latitude: 11.9416, longitude: 79.8083 },
  jodhpur:     { name: "Jodhpur",     country: "India", latitude: 26.2389, longitude: 73.0243 },
  hampi:       { name: "Hampi",       country: "India", latitude: 15.335,  longitude: 76.46   },
  // SRM campuses
  srm:         { name: "SRMIST Kattankulathur", country: "India", latitude: 12.8231, longitude: 80.0442 },
  srmist:      { name: "SRMIST Kattankulathur", country: "India", latitude: 12.8231, longitude: 80.0442 },
  kattankulathur: { name: "Kattankulathur", country: "India", latitude: 12.8231, longitude: 80.0442 },
};

export async function geocodePlace(name) {
  if (!name) return null;
  const norm = String(name).trim().toLowerCase();
  for (const [k, v] of Object.entries(KNOWN_COORDS)) {
    if (norm.includes(k) || k.includes(norm)) return { ...v };
  }
  try {
    const url = new URL("https://geocoding-api.open-meteo.com/v1/search");
    url.searchParams.set("name", name);
    url.searchParams.set("count", "1");
    url.searchParams.set("format", "json");
    const res = await fetch(url, { signal: AbortSignal.timeout?.(3000) });
    if (!res.ok) return null;
    const data = await res.json();
    const f = data.results?.[0];
    if (!f) return null;
    return { name: f.name, country: f.country, latitude: f.latitude, longitude: f.longitude };
  } catch { return null; }
}

export async function fetchWeather(lat, lon, days = 5) {
  if (typeof lat !== "number" || typeof lon !== "number") return [];
  try {
    const url = new URL("https://api.open-meteo.com/v1/forecast");
    url.searchParams.set("latitude", String(lat));
    url.searchParams.set("longitude", String(lon));
    url.searchParams.set("daily", "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,wind_speed_10m_max");
    url.searchParams.set("timezone", "auto");
    url.searchParams.set("forecast_days", String(Math.min(days, 7)));
    const res = await fetch(url, { signal: AbortSignal.timeout?.(4000) });
    if (!res.ok) return [];
    const data = await res.json();
    const d = data.daily;
    if (!d) return [];
    return d.time.map((date, i) => ({
      date, label: `Day ${i + 1}`,
      weatherCode: d.weather_code[i],
      max: Math.round(d.temperature_2m_max[i]),
      min: Math.round(d.temperature_2m_min[i]),
      precipitation: d.precipitation_probability_max?.[i] || 0,
      windSpeed: d.wind_speed_10m_max?.[i] || 0
    }));
  } catch { return []; }
}

export function weatherEmoji(code) {
  if ([0].includes(code)) return "☀️";
  if ([1, 2].includes(code)) return "🌤️";
  if ([3].includes(code)) return "☁️";
  if ([45, 48].includes(code)) return "🌫️";
  if ([51, 53, 55, 61, 63, 65, 80, 81, 82].includes(code)) return "🌧️";
  if ([71, 73, 75, 77, 85, 86].includes(code)) return "❄️";
  if ([95, 96, 99].includes(code)) return "⛈️";
  return "🌥️";
}

/* Curated REAL hotels per major Indian city — used by /api/hotels/search.
   (Subset of GitHub repo's REAL_HOTELS_BY_CITY for Workers footprint.) */
export const REAL_HOTELS_BY_CITY = {
  delhi: [
    { name: "Taj Palace, New Delhi",      area: "Diplomatic Enclave", stars: 5, basePrice: 14500, rating: 4.7, amenities: ["WiFi","Pool","Spa","Gym","Concierge"] },
    { name: "The Leela Palace New Delhi", area: "Chanakyapuri",       stars: 5, basePrice: 18500, rating: 4.8, amenities: ["WiFi","Pool","Spa","Butler","Gym"] },
    { name: "Radisson Blu Plaza Delhi",   area: "Mahipalpur",         stars: 5, basePrice: 7800,  rating: 4.4, amenities: ["WiFi","Pool","Gym","Airport Shuttle"] },
    { name: "Lemon Tree Premier",         area: "Aerocity",           stars: 4, basePrice: 5400,  rating: 4.3, amenities: ["WiFi","Pool","Gym","Restaurant"] },
    { name: "Treebo Trend Daksh",         area: "Paharganj",          stars: 3, basePrice: 1800,  rating: 3.9, amenities: ["WiFi","AC","Breakfast"] },
    { name: "Zostel Delhi (Hostel)",      area: "Paharganj",          stars: 2, basePrice: 700,   rating: 4.2, amenities: ["WiFi","Backpacker"] },
  ],
  mumbai: [
    { name: "The Taj Mahal Palace, Mumbai", area: "Colaba",        stars: 5, basePrice: 22500, rating: 4.8, amenities: ["WiFi","Pool","Spa","Heritage","Sea View"] },
    { name: "The Oberoi Mumbai",            area: "Nariman Point", stars: 5, basePrice: 21000, rating: 4.8, amenities: ["WiFi","Pool","Spa","Sea View"] },
    { name: "JW Marriott Mumbai Juhu",      area: "Juhu",          stars: 5, basePrice: 13800, rating: 4.6, amenities: ["WiFi","Pool","Spa","Beach Access"] },
    { name: "Lemon Tree Premier Mumbai",    area: "Andheri East",  stars: 4, basePrice: 6800,  rating: 4.3, amenities: ["WiFi","Pool","Gym"] },
    { name: "Treebo Trend Sea Pearl",       area: "Bandra West",   stars: 3, basePrice: 3200,  rating: 4.0, amenities: ["WiFi","AC","Breakfast"] },
    { name: "Zostel Mumbai (Hostel)",       area: "Andheri West",  stars: 2, basePrice: 850,   rating: 4.3, amenities: ["WiFi","Backpacker"] },
  ],
  bangalore: [
    { name: "The Leela Palace Bengaluru",  area: "Old Airport Road", stars: 5, basePrice: 13500, rating: 4.7, amenities: ["WiFi","Pool","Spa","Butler"] },
    { name: "ITC Gardenia",                 area: "Residency Road",   stars: 5, basePrice: 13800, rating: 4.7, amenities: ["WiFi","Pool","Spa","Gym"] },
    { name: "Lemon Tree Premier Ulsoor",    area: "Ulsoor",           stars: 4, basePrice: 5400,  rating: 4.3, amenities: ["WiFi","Pool","Gym"] },
    { name: "Treebo Trend Pearl Suites",    area: "Indiranagar",      stars: 3, basePrice: 2800,  rating: 4.1, amenities: ["WiFi","AC","Breakfast"] },
    { name: "Zostel Bangalore (Hostel)",    area: "Indiranagar",      stars: 2, basePrice: 750,   rating: 4.4, amenities: ["WiFi","Backpacker"] },
  ],
  jaipur: [
    { name: "Rambagh Palace, Jaipur (Taj)", area: "Bhawani Singh Rd", stars: 5, basePrice: 34500, rating: 4.9, amenities: ["WiFi","Pool","Heritage Palace"] },
    { name: "Jai Mahal Palace (Taj)",        area: "Civil Lines",      stars: 5, basePrice: 13800, rating: 4.7, amenities: ["WiFi","Pool","Heritage"] },
    { name: "Radisson Blu Jaipur",           area: "Tonk Road",        stars: 5, basePrice: 7800,  rating: 4.4, amenities: ["WiFi","Pool","Gym","Spa"] },
    { name: "Lemon Tree Premier Jaipur",     area: "Tonk Road",        stars: 4, basePrice: 5200,  rating: 4.3, amenities: ["WiFi","Pool","Gym"] },
    { name: "Treebo Trend Hari Mahal",       area: "Bani Park",        stars: 3, basePrice: 2500,  rating: 4.0, amenities: ["WiFi","AC","Breakfast"] },
    { name: "Zostel Jaipur (Hostel)",        area: "Bani Park",        stars: 2, basePrice: 650,   rating: 4.4, amenities: ["WiFi","Rooftop","Backpacker"] },
  ],
  goa: [
    { name: "Taj Exotica Resort & Spa",      area: "Benaulim",   stars: 5, basePrice: 18500, rating: 4.7, amenities: ["WiFi","Beach","Pool","Spa"] },
    { name: "The Leela Goa",                  area: "Cavelossim", stars: 5, basePrice: 17800, rating: 4.7, amenities: ["WiFi","Beach","Pool","Casino"] },
    { name: "Novotel Goa Resort & Spa",       area: "Candolim",   stars: 5, basePrice: 8500,  rating: 4.4, amenities: ["WiFi","Pool","Spa"] },
    { name: "Lemon Tree Amarante",            area: "Candolim",   stars: 4, basePrice: 5800,  rating: 4.3, amenities: ["WiFi","Pool","Beach Access"] },
    { name: "Treebo Trend Apollo Bay",        area: "Calangute",  stars: 3, basePrice: 2900,  rating: 4.0, amenities: ["WiFi","Pool","AC"] },
    { name: "Zostel Goa (Hostel)",            area: "Anjuna",     stars: 2, basePrice: 850,   rating: 4.5, amenities: ["WiFi","Pool","Beach"] },
  ],
  chennai: [
    { name: "ITC Grand Chola, Chennai",       area: "Guindy",       stars: 5, basePrice: 13500, rating: 4.7, amenities: ["WiFi","Pool","Spa"] },
    { name: "Taj Coromandel",                 area: "Nungambakkam", stars: 5, basePrice: 11800, rating: 4.7, amenities: ["WiFi","Pool","Spa"] },
    { name: "Hyatt Regency Chennai",          area: "Mount Road",   stars: 5, basePrice: 8800,  rating: 4.5, amenities: ["WiFi","Pool","Spa"] },
    { name: "Novotel Chennai OMR",            area: "OMR",          stars: 4, basePrice: 5800,  rating: 4.4, amenities: ["WiFi","Pool","Gym"] },
    { name: "Treebo Trend Adyar Gate",        area: "Adyar",        stars: 3, basePrice: 2400,  rating: 4.0, amenities: ["WiFi","AC","Breakfast"] },
    { name: "Zostel Chennai (Hostel)",        area: "Triplicane",   stars: 2, basePrice: 700,   rating: 4.3, amenities: ["WiFi","Backpacker"] },
  ],
  hyderabad: [
    { name: "Taj Falaknuma Palace",           area: "Falaknuma",    stars: 5, basePrice: 32000, rating: 4.9, amenities: ["WiFi","Pool","Heritage Palace"] },
    { name: "ITC Kohenur",                     area: "HITEC City",   stars: 5, basePrice: 11500, rating: 4.7, amenities: ["WiFi","Pool","Spa"] },
    { name: "Park Hyatt Hyderabad",            area: "Banjara Hills",stars: 5, basePrice: 11800, rating: 4.7, amenities: ["WiFi","Pool","Spa"] },
    { name: "Lemon Tree Premier HITEC City",   area: "HITEC City",   stars: 4, basePrice: 4900,  rating: 4.3, amenities: ["WiFi","Pool","Gym"] },
    { name: "Treebo Trend Hometel",            area: "Begumpet",     stars: 3, basePrice: 2300,  rating: 4.0, amenities: ["WiFi","AC","Breakfast"] },
  ],
  agra: [
    { name: "The Oberoi Amarvilas",            area: "Taj East Gate", stars: 5, basePrice: 42000, rating: 4.9, amenities: ["WiFi","Pool","Spa","Taj View"] },
    { name: "ITC Mughal",                       area: "Taj Ganj",      stars: 5, basePrice: 13800, rating: 4.7, amenities: ["WiFi","Pool","Kaya Kalp Spa"] },
    { name: "Trident Agra",                     area: "Fatehabad Road",stars: 5, basePrice: 9500,  rating: 4.5, amenities: ["WiFi","Pool","Gym"] },
    { name: "Treebo Trend Crystal Inn",         area: "Fatehabad Road",stars: 3, basePrice: 2200,  rating: 4.0, amenities: ["WiFi","AC","Taj View"] },
    { name: "Zostel Agra (Hostel)",             area: "Tajganj",       stars: 2, basePrice: 600,   rating: 4.4, amenities: ["WiFi","Rooftop"] },
  ],
  udaipur: [
    { name: "The Oberoi Udaivilas",             area: "Lake Pichola", stars: 5, basePrice: 48000, rating: 4.9, amenities: ["WiFi","Pool","Lake View"] },
    { name: "Taj Lake Palace, Udaipur",         area: "Lake Pichola", stars: 5, basePrice: 55000, rating: 4.9, amenities: ["WiFi","Heritage Palace"] },
    { name: "Trident Udaipur",                   area: "Haridasji Magri",stars:5,basePrice: 9800,  rating: 4.6, amenities: ["WiFi","Pool","Lake View"] },
    { name: "Treebo Trend Garden Hotel",        area: "Bhattiyani",   stars: 3, basePrice: 2200,  rating: 4.0, amenities: ["WiFi","AC","Breakfast"] },
  ],
  pondicherry: [
    { name: "The Promenade Pondicherry",        area: "Rock Beach",   stars: 5, basePrice: 9800,  rating: 4.5, amenities: ["WiFi","Pool","Sea View"] },
    { name: "Palais De Mahe",                    area: "White Town",   stars: 4, basePrice: 6500,  rating: 4.5, amenities: ["WiFi","Pool","Heritage"] },
    { name: "Lemon Tree Pondicherry",            area: "Mission Street",stars:4, basePrice: 4500,  rating: 4.2, amenities: ["WiFi","Pool","Gym"] },
    { name: "Treebo Trend Maison Radha",         area: "Goubert Ave",  stars: 3, basePrice: 2400,  rating: 4.0, amenities: ["WiFi","AC","Breakfast"] },
  ],
};
