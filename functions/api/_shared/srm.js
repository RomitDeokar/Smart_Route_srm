/* ════════════════════════════════════════════════════════════════
   _shared/srm.js — SRM-specific accommodation database & helpers,
   merged from GitHub src/index.tsx (RomitDeokar/Smart_Route_srm).

   Provides:
   • SRM_SPECIFIC_HOTELS — real on-campus & near-campus options
   • isSRMCity() — maps free-text destination → canonical SRM key
   ════════════════════════════════════════════════════════════════ */

export const SRM_SPECIFIC_HOTELS = {
  chennai: [
    {name:'SRM Hotel (Maamallan)',stars:4,basePrice:3200,rating:4.3,
     amenities:['WiFi','AC','Breakfast','Restaurant','Conference Hall','Parking','Near SRM KTR'],
     address:'Potheri, Kattankulathur, Chennai',
     description:'Official 4-star hotel run by SRM Group, walking distance from SRMIST main campus. Preferred for parents and visiting faculty.',
     officialUrl:'https://srmhotels.com/',
     applyRequired:false,
     srmOfficial:true,
     image:'https://srmhotels.com/wp-content/uploads/2020/11/srm-hotel-front.jpg'},
    {name:'Premium Boys Hostel (SRMIST)',stars:4,basePrice:0,rating:4.5,
     amenities:['AC Rooms','WiFi','Mess','Laundry','Gym','24x7 Security','On-Campus','Reading Room'],
     address:'SRMIST Kattankulathur Campus, Chennai',
     description:'On-campus premium accommodation for SRMIST male students. Allocation by application only — click "Apply for Hostel" to submit your request.',
     officialUrl:'https://www.srmist.edu.in/hostels/',
     applyRequired:true,
     applyUrl:'https://www.srmist.edu.in/hostels/',
     hostelType:'boys',
     srmOfficial:true,
     image:''},
    {name:'Premium Girls Hostel (SRMIST)',stars:4,basePrice:0,rating:4.5,
     amenities:['AC Rooms','WiFi','Mess','Laundry','Gym','24x7 Security','On-Campus','Reading Room'],
     address:'SRMIST Kattankulathur Campus, Chennai',
     description:'On-campus premium accommodation for SRMIST female students. Allocation by application only — click "Apply for Hostel" to submit your request.',
     officialUrl:'https://www.srmist.edu.in/hostels/',
     applyRequired:true,
     applyUrl:'https://www.srmist.edu.in/hostels/',
     hostelType:'girls',
     srmOfficial:true,
     image:''},
    {name:'GRT Grand Days (Near SRM)',stars:3,basePrice:2200,rating:4.0,
     amenities:['WiFi','AC','Breakfast','Restaurant','Parking'],
     address:'Guduvancheri, near SRM Kattankulathur',
     description:'Comfortable 3-star option close to SRMIST, popular with parents and corporate visitors.',
     image:''},
    {name:'Hotel Turyaa Chennai (Old Mahabalipuram Rd)',stars:4,basePrice:4500,rating:4.2,
     amenities:['WiFi','AC','Pool','Breakfast','Gym','Restaurant'],
     address:'OMR, Chennai (~25km from SRM KTR)',
     description:'Modern 4-star property on OMR; convenient for SRM-related conferences and events.',
     image:''},
  ],
  trichy: [
    {name:'SRM Hotel Trichy (on-campus guest house)',stars:3,basePrice:2400,rating:4.1,
     amenities:['WiFi','AC','Mess','Parking','On-Campus','Breakfast'],
     address:'SRM Trichy Campus, Tiruchirappalli',
     description:'On-campus guest house at SRM Trichy. Ideal for visiting parents and academics.',
     officialUrl:'https://www.srmtrichy.edu.in/',
     applyRequired:false,
     srmOfficial:true,
     image:''},
    {name:'SRM Trichy Boys Hostel',stars:3,basePrice:0,rating:4.2,
     amenities:['WiFi','Mess','Laundry','24x7 Security','On-Campus'],
     address:'SRM Trichy Campus',
     description:'On-campus hostel for SRM Trichy male students. Apply for allocation through the hostel office.',
     officialUrl:'https://www.srmtrichy.edu.in/',
     applyRequired:true,
     applyUrl:'https://www.srmtrichy.edu.in/',
     hostelType:'boys',
     srmOfficial:true,
     image:''},
  ],
  'delhi ncr': [
    {name:'SRM University Delhi-NCR Guest House',stars:3,basePrice:2800,rating:4.0,
     amenities:['WiFi','AC','Breakfast','On-Campus','Parking'],
     address:'SRM NCR Campus, Modinagar',
     description:'Official guest house at SRM Delhi-NCR campus.',
     officialUrl:'https://www.srmuniversity.ac.in/',
     applyRequired:false,
     srmOfficial:true,
     image:''},
    {name:'SRM NCR Boys Hostel',stars:3,basePrice:0,rating:4.1,
     amenities:['WiFi','Mess','Laundry','24x7 Security','On-Campus'],
     address:'SRM NCR Campus, Modinagar',
     description:'On-campus hostel for SRM NCR male students. Apply via hostel office.',
     officialUrl:'https://www.srmuniversity.ac.in/',
     applyRequired:true,
     applyUrl:'https://www.srmuniversity.ac.in/',
     hostelType:'boys',
     srmOfficial:true,
     image:''},
  ],
  amaravati: [
    {name:'SRM AP University Guest House',stars:3,basePrice:2600,rating:4.0,
     amenities:['WiFi','AC','On-Campus','Mess','Parking'],
     address:'SRM AP Campus, Amaravati',
     description:'On-campus guest house at SRM Andhra Pradesh.',
     officialUrl:'https://srmap.edu.in/',
     applyRequired:false,
     srmOfficial:true,
     image:''},
    {name:'SRM AP Boys Hostel',stars:3,basePrice:0,rating:4.2,
     amenities:['AC','WiFi','Mess','Laundry','On-Campus','24x7 Security'],
     address:'SRM AP Campus, Amaravati',
     description:'Premium on-campus hostel at SRM AP. Apply through the campus hostel office.',
     officialUrl:'https://srmap.edu.in/',
     applyRequired:true,
     applyUrl:'https://srmap.edu.in/',
     hostelType:'boys',
     srmOfficial:true,
     image:''},
  ],
  sikkim: [
    {name:'SRM Sikkim Guest House',stars:3,basePrice:2200,rating:3.9,
     amenities:['WiFi','Heating','Mess','On-Campus','Parking'],
     address:'SRM Sikkim Campus, Gangtok',
     description:'On-campus guest house at SRM Sikkim, with heating for cold-weather stays.',
     officialUrl:'https://srmsikkim.edu.in/',
     applyRequired:false,
     srmOfficial:true,
     image:''},
    {name:'SRM Sikkim Boys Hostel',stars:3,basePrice:0,rating:4.0,
     amenities:['WiFi','Mess','Heating','24x7 Security','On-Campus'],
     address:'SRM Sikkim Campus, Gangtok',
     description:'On-campus hostel for SRM Sikkim male students.',
     officialUrl:'https://srmsikkim.edu.in/',
     applyRequired:true,
     applyUrl:'https://srmsikkim.edu.in/',
     hostelType:'boys',
     srmOfficial:true,
     image:''},
  ],
};

/**
 * Map free-text destination → canonical SRM hotel-list key.
 * Returns one of: 'chennai' | 'trichy' | 'delhi ncr' | 'amaravati' | 'sikkim' | null
 *
 * Bug-fix from GitHub: the original used `||` and `&&` without grouping, which caused
 * JS operator-precedence pitfalls. Rewritten with explicit checks + dedicated regional
 * keywords so that searches like "SRM", "SRMIST", "Kattankulathur", "Chennai",
 * "SRM KTR", "SRM Trichy", "SRM NCR", "SRM AP", "SRM Sikkim" all reliably surface
 * SRM-specific accommodations.
 */
export function isSRMCity(city) {
  const k = String(city || '').toLowerCase().trim();
  if (!k) return null;

  // Trichy campus
  if (k.includes('trichy') || k.includes('tiruchirappalli') || /\bsrm\s*trichy\b/.test(k)) return 'trichy';

  // Delhi-NCR campus
  if (k.includes('delhi ncr') || k.includes('delhi-ncr') || k.includes('greater noida') ||
      k.includes('modinagar') || /\bsrm\s*ncr\b/.test(k) || /\bsrm\s*delhi\b/.test(k)) return 'delhi ncr';

  // Amaravati / AP campus
  if (k.includes('amaravati') || k.includes('vijayawada') || /\bsrm\s*ap\b/.test(k) ||
      /\bsrm\s*andhra\b/.test(k)) return 'amaravati';

  // Sikkim campus
  if (/\bsrm\s*sikkim\b/.test(k) || (k.includes('gangtok') && /srm/.test(k))) return 'sikkim';

  // Chennai / Kattankulathur (default SRM main campus)
  const isChennai = k.includes('chennai');
  const isKattankulathur = k.includes('kattankulathur') || k.includes('katankulathur');
  const isMamallapuram = k.includes('mamallapur') || k.includes('mahabalipuram');
  const isSrmKeyword = /\bsrm\b/.test(k) || /\bsrmist\b/.test(k) ||
                       /\bsrm\s*ktr\b/.test(k) || /\bsrm\s*main\b/.test(k);
  const isExactSrm = k === 'srm' || k === 'srmist';
  if (isChennai || isKattankulathur || isMamallapuram || isSrmKeyword || isExactSrm) return 'chennai';

  return null;
}
