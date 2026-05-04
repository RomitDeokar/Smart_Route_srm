/* POST /api/hotels/search — real hotel chains + SRM-specific accommodations. */

import { jsonResponse } from "../_shared/auth.js";
import { REAL_HOTELS_BY_CITY } from "../_shared/data.js";
import { isSRMCity, SRM_SPECIFIC_HOTELS } from "../_shared/srm.js";

export const onRequestOptions = () => jsonResponse({ ok: true });

export const onRequestPost = async ({ request }) => {
  let body = {};
  try { body = await request.json(); } catch { body = {}; }
  const { city, budget, check_in, check_out } = body || {};

  if (!city || !String(city).trim()) {
    return jsonResponse({ ok: false, error: "City is required." }, 400);
  }

  const cityKey = String(city).toLowerCase().trim().replace(/[^a-z\s]/g, "");

  // Find curated REAL hotel list
  let hotelList = [];
  for (const [k, list] of Object.entries(REAL_HOTELS_BY_CITY)) {
    if (cityKey.includes(k) || k.includes(cityKey)) { hotelList = [...list]; break; }
  }
  if (!hotelList.length) {
    hotelList = [
      { name: `Taj ${city}`,            area: "City Center",       stars: 5, basePrice: 9800, rating: 4.6, amenities: ["WiFi","Pool","Spa","Restaurant","Gym"] },
      { name: `ITC ${city}`,             area: "Business District", stars: 5, basePrice: 8500, rating: 4.5, amenities: ["WiFi","Pool","Spa","Restaurant"] },
      { name: `Radisson Blu ${city}`,    area: "Central",           stars: 5, basePrice: 7200, rating: 4.4, amenities: ["WiFi","Pool","Gym","Spa"] },
      { name: `Lemon Tree ${city}`,      area: "City Center",       stars: 4, basePrice: 4500, rating: 4.2, amenities: ["WiFi","Pool","Gym"] },
      { name: `Treebo Trend ${city}`,    area: "Central",           stars: 3, basePrice: 2200, rating: 4.0, amenities: ["WiFi","AC","Breakfast"] },
      { name: `OYO Townhouse ${city}`,   area: "Central",           stars: 3, basePrice: 1500, rating: 3.7, amenities: ["WiFi","AC","Breakfast"] },
      { name: `Zostel ${city} (Hostel)`, area: "Central",           stars: 2, basePrice: 650,  rating: 4.3, amenities: ["WiFi","Backpacker"] },
    ];
  }

  // Inject SRM-specific hotels at the top if destination is an SRM campus
  const srmKey = isSRMCity(city);
  const srmList = srmKey ? (SRM_SPECIFIC_HOTELS[srmKey] || []) : [];

  const checkinDate  = check_in  || new Date().toISOString().split("T")[0];
  const checkoutDate = check_out || new Date(Date.now() + 86400000).toISOString().split("T")[0];
  const cityEnc = encodeURIComponent(city);
  const searchUrl = `https://www.booking.com/searchresults.html?ss=${cityEnc}&checkin=${checkinDate}&checkout=${checkoutDate}&group_adults=2&no_rooms=1`;

  const days = Math.max(1, Math.ceil(
    (new Date(checkoutDate) - new Date(checkinDate)) / 86400000
  ));

  const all = [...srmList, ...hotelList];
  const hotels = all.map((h, i) => {
    const variation = 0.9 + ((i * 7) % 20) / 100;
    const ppn = Math.round((h.basePrice || 0) * variation);
    const isHostel = !!h.applyRequired;
    const platforms = h.srmOfficial && h.officialUrl
      ? [{ name: isHostel ? "Apply on SRMIST Portal" : "SRM Official",
           url: h.officialUrl, prefilled: true, srmOfficial: true }]
      : [];
    if (!isHostel) {
      platforms.push(
        { name: "Booking.com", url: searchUrl, prefilled: true },
        { name: "MakeMyTrip", url: `https://www.makemytrip.com/hotels/hotel-listing?city=${cityEnc}`, prefilled: true },
        { name: "Goibibo",    url: `https://www.goibibo.com/hotels/hotels-in-${city.toLowerCase().replace(/\s+/g,"-")}/`, prefilled: true },
        { name: "Agoda",      url: `https://www.agoda.com/search?city=${cityEnc}`, prefilled: true },
        { name: "OYO",        url: `https://www.oyorooms.com/search?location=${cityEnc}`, prefilled: true },
      );
    }
    return {
      id: `HT${(h.name || "X").replace(/\s+/g, "").slice(0, 8)}${i}`,
      name: h.name,
      chain: h.name.split(" ")[0],
      stars: h.stars,
      pricePerNight: ppn,
      price_per_night: ppn,
      total_price: ppn * days,
      priceFormatted: `₹${ppn.toLocaleString("en-IN")}`,
      rating: (typeof h.rating === "number" ? h.rating : 4.0).toFixed(1),
      reviewCount: 200 + ((i * 137) % 2800),
      amenities: h.amenities,
      address: h.address || (h.area ? `${h.area}, ${city}` : city),
      distanceFromCenter: `${(0.5 + (i * 0.7) % 5).toFixed(1)} km`,
      description: h.description || "",
      bookingUrl: isHostel ? (h.applyUrl || h.officialUrl || "#") : searchUrl,
      bookingPlatforms: platforms,
      cancellationPolicy: i % 3 === 0 ? "Free cancellation" : "Non-refundable",
      // SRM-specific flags consumed by the frontend
      srmOfficial: !!h.srmOfficial,
      applyRequired: !!h.applyRequired,
      applyUrl: h.applyUrl || "",
      hostelType: h.hostelType || "",
    };
  }).sort((a, b) => {
    if (a.srmOfficial && !b.srmOfficial) return -1;
    if (!a.srmOfficial && b.srmOfficial) return 1;
    return a.pricePerNight - b.pricePerNight;
  });

  return jsonResponse({ ok: true, hotels, bookingUrl: searchUrl });
};
