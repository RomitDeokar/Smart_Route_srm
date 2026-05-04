import { useState } from "react";
import { motion } from "framer-motion";

/* ══════════════════════════════════════════════════════════
   LANGUAGE DATABASE — covers cities, districts & regions
   Detection: city name → state → language(s)
   ══════════════════════════════════════════════════════════ */

/* Map keywords → language key */
const KEYWORD_TO_LANG = {
  // Tamil Nadu (Tamil)
  coorg:"kannada",
  chennai:"tamil", madurai:"tamil", coimbatore:"tamil", trichy:"tamil",
  tiruchirappalli:"tamil", salem:"tamil", tirunelveli:"tamil", vellore:"tamil",
  erode:"tamil", thoothukudi:"tamil", ooty:"tamil", kodaikanal:"tamil",
  kanchipuram:"tamil", pondicherry:"tamil", puducherry:"tamil", thanjavur:"tamil",
  cuddalore:"tamil", nagapattinam:"tamil", karur:"tamil", dindigul:"tamil",

  // Kerala (Malayalam)
  munnar:"malayalam", varkala:"malayalam", kovalam:"malayalam", alleppey:"malayalam",
  kochi:"malayalam", thiruvananthapuram:"malayalam", trivandrum:"malayalam",
  kozhikode:"malayalam", calicut:"malayalam", thrissur:"malayalam",
  kollam:"malayalam", kannur:"malayalam", palakkad:"malayalam",
  alappuzha:"malayalam",
  wayanad:"malayalam", idukki:"malayalam", kottayam:"malayalam",

  // Karnataka (Kannada)
  bangalore:"kannada", bengaluru:"kannada", mysore:"kannada", mysuru:"kannada",
  hubli:"kannada", dharwad:"kannada", mangalore:"kannada", belgaum:"kannada",
  hampi:"kannada", badami:"kannada", udupi:"kannada", gulbarga:"kannada",
  hassan:"kannada", shimoga:"kannada", tumkur:"kannada",

  // Andhra Pradesh & Telangana (Telugu)
  hyderabad:"telugu", visakhapatnam:"telugu", vizag:"telugu",
  vijayawada:"telugu", guntur:"telugu", tirupati:"telugu",
  nellore:"telugu", kurnool:"telugu", rajahmundry:"telugu",
  warangal:"telugu", karimnagar:"telugu", nizamabad:"telugu",

  // West Bengal (Bengali)
  kolkata:"bengali", calcutta:"bengali", darjeeling:"bengali",
  siliguri:"bengali", durgapur:"bengali", asansol:"bengali",
  howrah:"bengali", burdwan:"bengali", malda:"bengali",
  murshidabad:"bengali", bankura:"bengali", purulia:"bengali",

  // Maharashtra (Marathi)
  mumbai:"marathi", pune:"marathi", nagpur:"marathi", nashik:"marathi",
  aurangabad:"marathi", solapur:"marathi", kolhapur:"marathi",
  thane:"marathi", navi:"marathi", lonavala:"marathi", mahabaleshwar:"marathi",
  shirdi:"marathi", ajanta:"marathi", ellora:"marathi",

  // Gujarat (Gujarati)
  ahmedabad:"gujarati", surat:"gujarati", vadodara:"gujarati", baroda:"gujarati",
  rajkot:"gujarati", gandhinagar:"gujarati", bhavnagar:"gujarati",
  junagadh:"gujarati", jamnagar:"gujarati", rann:"gujarati", kutch:"gujarati",

  // Rajasthan (Rajasthani/Hindi)
  jaipur:"rajasthani", jodhpur:"rajasthani", udaipur:"rajasthani",
  jaisalmer:"rajasthani", bikaner:"rajasthani", pushkar:"rajasthani",
  ajmer:"rajasthani", chittorgarh:"rajasthani", mount:"rajasthani",

  // Uttar Pradesh (Hindi)
  agra:"hindi_up", varanasi:"hindi_up", lucknow:"hindi_up",
  allahabad:"hindi_up", prayagraj:"hindi_up", kanpur:"hindi_up",
  mathura:"hindi_up", vrindavan:"hindi_up", ayodhya:"hindi_up",
  gorakhpur:"hindi_up", meerut:"hindi_up",

  // Uttarakhand (Hindi/Garhwali)
  rishikesh:"garhwali", haridwar:"garhwali", dehradun:"garhwali",
  mussoorie:"garhwali", nainital:"garhwali", auli:"garhwali",

  // Himachal Pradesh (Hindi/Pahari)
  shimla:"pahari", manali:"pahari", dharamshala:"pahari",
  mcleod:"pahari", kullu:"pahari", spiti:"pahari", kinnaur:"pahari",

  // Delhi (Hindi)
  delhi:"hindi_delhi", "new delhi":"hindi_delhi", noida:"hindi_delhi", gurgaon:"hindi_delhi",

  // Goa (Konkani)
  goa:"konkani", panaji:"konkani", margao:"konkani", vasco:"konkani",

  // Punjab (Punjabi)
  amritsar:"punjabi", ludhiana:"punjabi", chandigarh:"punjabi",
  jalandhar:"punjabi", patiala:"punjabi", golden:"punjabi",

  // Northeast (various)
  shillong:"khasi", guwahati:"assamese", dispur:"assamese",
  imphal:"manipuri", kohima:"nagamese", aizawl:"mizo",
  gangtok:"nepali", itanagar:"hindi_ne",

  // Jammu & Kashmir
  srinagar:"kashmiri", jammu:"kashmiri", leh:"ladakhi", ladakh:"ladakhi",

  // Odisha (Odia)
  bhubaneswar:"odia", puri:"odia", cuttack:"odia", konark:"odia",

  // Madhya Pradesh (Hindi)
  bhopal:"hindi_mp", indore:"hindi_mp", gwalior:"hindi_mp",
  jabalpur:"hindi_mp", khajuraho:"hindi_mp", ujjain:"hindi_mp",

  // Bihar (Bhojpuri/Hindi)
  patna:"bhojpuri", bodh:"bhojpuri", gaya:"bhojpuri",

  // Assam
  kaziranga:"assamese", majuli:"assamese", jorhat:"assamese",

  // Generic beach/hill
  andaman:"bengali",
};

const LANG_DATA = {
  tamil: {
    language: "Tamil", script: "தமிழ்",
    phrases: [
      { local:"வணக்கம் (Vanakkam)",    meaning:"Hello / Greetings",    phonetic:"Va-na-kam" },
      { local:"நன்றி (Nandri)",         meaning:"Thank you",            phonetic:"Nan-dri" },
      { local:"விலை என்ன? (Vilai enna?)",meaning:"What is the price?",   phonetic:"Vi-lai en-na?" },
      { local:"சாப்பிட்டீர்களா?",       meaning:"Have you eaten?",      phonetic:"Sap-pit-teer-ga-la?" },
      { local:"புரியவில்லை",           meaning:"I don't understand",   phonetic:"Pu-ri-ya-vil-lai" },
      { local:"நல்லா இருக்கீங்களா?",   meaning:"Are you doing well?",  phonetic:"Nal-la i-ruk-kee-nga-la?" },
      { local:"எங்கே? (Enge?)",         meaning:"Where is it?",         phonetic:"En-gay?" },
      { local:"ஆமா / இல்லை",           meaning:"Yes / No",            phonetic:"Aa-ma / Il-lai" },
    ],
  },

  malayalam: {
    language: "Malayalam", script: "മലയാളം",
    phrases: [
      { local:"നമസ്കാരം (Namaskaram)", meaning:"Hello (respectful)",   phonetic:"Na-mas-ka-ram" },
      { local:"നന്ദി (Nandi)",          meaning:"Thank you",            phonetic:"Nan-dee" },
      { local:"സുഖമാണോ? (Sukhamano?)", meaning:"Are you well?",        phonetic:"Soo-kha-ma-no?" },
      { local:"വില എത്ര? (Vila ethra?)",meaning:"What is the price?",  phonetic:"Vi-la et-hra?" },
      { local:"സഹായിക്കാമോ?",           meaning:"Can you help?",        phonetic:"Sa-ha-yi-kka-mo?" },
      { local:"എനിക്ക് മലയാളം അറിയില്ല", meaning:"I don't know Malayalam", phonetic:"E-nik-ku a-ri-yil-la" },
      { local:"ഇടത്ത് / വലത്ത്",        meaning:"Left / Right",         phonetic:"I-dat-tu / Va-lat-tu" },
      { local:"ഹാ / ഇല്ല",              meaning:"Yes / No",            phonetic:"Haa / Il-la" },
    ],
  },

  kannada: {
    language: "Kannada", script: "ಕನ್ನಡ",
    phrases: [
      { local:"ನಮಸ್ಕಾರ (Namaskara)",    meaning:"Hello",                phonetic:"Na-mas-ka-ra" },
      { local:"ಧನ್ಯವಾದಗಳು",            meaning:"Thank you",            phonetic:"Dhan-ya-va-da-ga-lu" },
      { local:"ಹೇಗಿದ್ದೀರಾ? (Hegiddeera?)", meaning:"How are you?",    phonetic:"He-gi-dee-ra?" },
      { local:"ಬೆಲೆ ಎಷ್ಟು? (Bela eshtu?)", meaning:"What is the price?", phonetic:"Be-la esh-tu?" },
      { local:"ನನಗೆ ಗೊತ್ತಿಲ್ಲ",         meaning:"I don't know",         phonetic:"Na-na-ge got-til-la" },
      { local:"ತುಂಬಾ ಚೆಂದ (Tumba chenda)", meaning:"Very beautiful",   phonetic:"Tum-ba chen-da" },
      { local:"ಎಡ / ಬಲ",               meaning:"Left / Right",         phonetic:"E-da / Ba-la" },
      { local:"ಹೌದು / ಇಲ್ಲ",           meaning:"Yes / No",            phonetic:"How-du / Il-la" },
    ],
  },

  telugu: {
    language: "Telugu", script: "తెలుగు",
    phrases: [
      { local:"నమస్కారం (Namaskaram)",  meaning:"Hello",                phonetic:"Na-mas-ka-ram" },
      { local:"ధన్యవాదాలు (Dhanyavaadalu)", meaning:"Thank you",       phonetic:"Dhan-ya-va-da-lu" },
      { local:"ఎలా ఉన్నారు?",           meaning:"How are you?",         phonetic:"E-la un-na-ru?" },
      { local:"ఎంత? (Entha?)",          meaning:"How much?",            phonetic:"En-tha?" },
      { local:"అర్థం కాలేదు",          meaning:"I don't understand",   phonetic:"Ar-tham ka-le-du" },
      { local:"ఎక్కడ? (Ekkada?)",       meaning:"Where?",               phonetic:"Ek-ka-da?" },
      { local:"అవును / కాదు",           meaning:"Yes / No",            phonetic:"A-vu-nu / Ka-du" },
      { local:"సహాయం చేయగలరా?",        meaning:"Can you help?",        phonetic:"Sa-ha-yam che-ya-ga-la-ra?" },
    ],
  },

  bengali: {
    language: "Bengali", script: "বাংলা",
    phrases: [
      { local:"নমস্কার (Namaskar)",     meaning:"Hello (formal)",       phonetic:"No-mos-kar" },
      { local:"আপনি কেমন আছেন?",       meaning:"How are you?",         phonetic:"Ap-ni ke-mon a-chen?" },
      { local:"ধন্যবাদ (Dhanyabad)",   meaning:"Thank you",            phonetic:"Dhon-no-bad" },
      { local:"কত দাম? (Koto dam?)",   meaning:"How much?",            phonetic:"Ko-to dam?" },
      { local:"বুঝতে পারছি না",        meaning:"I don't understand",   phonetic:"Buj-te par-chi na" },
      { local:"কোথায়? (Kothay?)",      meaning:"Where?",               phonetic:"Ko-thay?" },
      { local:"হ্যাঁ / না (Hyan / Na)", meaning:"Yes / No",            phonetic:"Hyan / Na" },
      { local:"সাহায্য করুন",          meaning:"Please help me",       phonetic:"Sa-ha-jo ko-run" },
    ],
  },

  marathi: {
    language: "Marathi", script: "मराठी",
    phrases: [
      { local:"नमस्कार (Namaskar)",     meaning:"Hello",                phonetic:"Na-mas-kar" },
      { local:"धन्यवाद (Dhanyavad)",   meaning:"Thank you",            phonetic:"Dhan-ya-vad" },
      { local:"कसे आहात? (Kase aahat?)", meaning:"How are you?",       phonetic:"Ka-se aa-hat?" },
      { local:"किती रुपये? (Kiti rupaye?)", meaning:"How much?",       phonetic:"Ki-ti ru-pa-ye?" },
      { local:"मला समजत नाही",         meaning:"I don't understand",   phonetic:"Ma-la sa-ma-jat na-hi" },
      { local:"कुठे? (Kuthe?)",         meaning:"Where?",               phonetic:"Ku-the?" },
      { local:"हो / नाही (Ho / Nahi)", meaning:"Yes / No",            phonetic:"Ho / Na-hi" },
      { local:"मदत करा",               meaning:"Please help",          phonetic:"Ma-dat ka-ra" },
    ],
  },

  gujarati: {
    language: "Gujarati", script: "ગુજરાતી",
    phrases: [
      { local:"જય શ્રી કૃષ્ણ",        meaning:"Hello (traditional)",  phonetic:"Jai Shree Krish-na" },
      { local:"ધન્યવાદ (Dhanyavad)",  meaning:"Thank you",            phonetic:"Dhan-ya-vad" },
      { local:"કેમ છો? (Kem cho?)",   meaning:"How are you?",         phonetic:"Kem cho?" },
      { local:"કેટલા? (Ketla?)",      meaning:"How much?",            phonetic:"Ket-la?" },
      { local:"મને નથી ખબર",          meaning:"I don't know",         phonetic:"Ma-ne na-thi kha-bar" },
      { local:"ક્યાં? (Kyan?)",       meaning:"Where?",               phonetic:"Kyan?" },
      { local:"હા / ના (Ha / Na)",    meaning:"Yes / No",            phonetic:"Haa / Naa" },
      { local:"મદદ કરો",              meaning:"Please help",          phonetic:"Ma-dad ka-ro" },
    ],
  },

  rajasthani: {
    language: "Rajasthani / Hindi", script: "राजस्थानी",
    phrases: [
      { local:"खम्मा घणी (Khamma Ghani)", meaning:"Blessings / Hello", phonetic:"Kham-ma Gha-ni" },
      { local:"पधारो म्हारे देश",        meaning:"Welcome to our land", phonetic:"Pa-dha-ro Mha-re Desh" },
      { local:"मेहरबानी (Meharbani)",    meaning:"Thank you / Kindness", phonetic:"Me-har-ba-ni" },
      { local:"सा जी (Saa ji)",          meaning:"Yes (respectful)",    phonetic:"Saa ji" },
      { local:"कितनो? (Kitno?)",         meaning:"How much?",           phonetic:"Kit-no?" },
      { local:"बहुत सुंदर (Bahut sundar)", meaning:"Very beautiful",  phonetic:"Ba-hut sun-dar" },
      { local:"आओ जी (Aao ji)",          meaning:"Please come / Welcome", phonetic:"Aa-o ji" },
      { local:"हाँ / ना",               meaning:"Yes / No",            phonetic:"Haan / Naa" },
    ],
  },

  hindi_up: {
    language: "Hindi (Awadhi/Bhojpuri mix)", script: "हिन्दी",
    phrases: [
      { local:"नमस्ते (Namaste)",       meaning:"Hello",                phonetic:"Na-mas-tay" },
      { local:"धन्यवाद (Dhanyavad)",   meaning:"Thank you",            phonetic:"Dhan-ya-vaad" },
      { local:"क्या हाल है?",          meaning:"How are you?",         phonetic:"Kya haal hai?" },
      { local:"कितना है? (Kitna hai?)", meaning:"How much?",           phonetic:"Kit-na hai?" },
      { local:"मुझे समझ नहीं आया",    meaning:"I don't understand",   phonetic:"Mu-jhe sa-majh na-hin aa-ya" },
      { local:"कहाँ है? (Kahan hai?)", meaning:"Where is it?",         phonetic:"Ka-han hai?" },
      { local:"हाँ / नहीं",           meaning:"Yes / No",            phonetic:"Haan / Na-hin" },
      { local:"मुझे मदद चाहिए",       meaning:"I need help",          phonetic:"Mu-jhe ma-dad cha-hi-ye" },
    ],
  },

  hindi_delhi: {
    language: "Hindi (Delhi)", script: "हिन्दी",
    phrases: [
      { local:"नमस्ते (Namaste)",       meaning:"Hello",                phonetic:"Na-mas-tay" },
      { local:"भाई / दीदी",            meaning:"Brother / Sister (friendly)", phonetic:"Bha-i / Di-di" },
      { local:"शुक्रिया (Shukriya)",   meaning:"Thank you",            phonetic:"Shuk-ri-ya" },
      { local:"कितने का है?",          meaning:"How much is it?",      phonetic:"Kit-ne ka hai?" },
      { local:"यार, ठीक कर दो",       meaning:"Come on, give me a deal", phonetic:"Yaar, theek kar do" },
      { local:"अरे यार! (Are yaar!)",  meaning:"Oh man! (surprise)",   phonetic:"A-re yaar!" },
      { local:"हाँ बिल्कुल",          meaning:"Yes, absolutely",       phonetic:"Haan bil-kul" },
      { local:"मेट्रो कहाँ है?",      meaning:"Where is the metro?",  phonetic:"Met-ro ka-han hai?" },
    ],
  },

  garhwali: {
    language: "Hindi / Garhwali", script: "हिन्दी",
    phrases: [
      { local:"नमस्ते (Namaste)",       meaning:"Hello",                phonetic:"Na-mas-tay" },
      { local:"धन्यवाद (Dhanyavad)",   meaning:"Thank you",            phonetic:"Dhan-ya-vaad" },
      { local:"बहुत ठंडा है",          meaning:"It's very cold",       phonetic:"Ba-hut than-da hai" },
      { local:"कितना दूर है?",         meaning:"How far is it?",       phonetic:"Kit-na dur hai?" },
      { local:"चाय मिलेगी?",           meaning:"Can I get tea?",       phonetic:"Chai mi-le-gi?" },
      { local:"यहाँ फोटो ले सकता हूँ?", meaning:"Can I take photos?", phonetic:"Ya-han pho-to le sak-ta hoon?" },
      { local:"हाँ / नहीं",           meaning:"Yes / No",            phonetic:"Haan / Na-hin" },
      { local:"राफ्टिंग कहाँ है?",    meaning:"Where is rafting?",    phonetic:"Raft-ing ka-han hai?" },
    ],
  },

  pahari: {
    language: "Hindi / Pahari", script: "हिन्दी",
    phrases: [
      { local:"नमस्ते (Namaste)",       meaning:"Hello",                phonetic:"Na-mas-tay" },
      { local:"धन्यवाद (Dhanyavad)",   meaning:"Thank you",            phonetic:"Dhan-ya-vaad" },
      { local:"बहुत सुंदर (Bahut sundar)", meaning:"Very beautiful",  phonetic:"Ba-hut sun-dar" },
      { local:"बर्फ कब गिरती है?",    meaning:"When does it snow?",   phonetic:"Barf kab gir-ti hai?" },
      { local:"रास्ता बंद है?",        meaning:"Is the road blocked?", phonetic:"Ras-ta band hai?" },
      { local:"होमस्टे मिलेगा?",       meaning:"Can I get a homestay?", phonetic:"Hom-stay mi-le-ga?" },
      { local:"हाँ / नहीं",           meaning:"Yes / No",            phonetic:"Haan / Na-hin" },
      { local:"दर्रा कितना ऊँचा है?", meaning:"How high is the pass?", phonetic:"Dar-ra kit-na oon-cha hai?" },
    ],
  },

  punjabi: {
    language: "Punjabi", script: "ਪੰਜਾਬੀ",
    phrases: [
      { local:"ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ (Sat Sri Akal)", meaning:"Hello (Sikh greeting)", phonetic:"Sat Sri A-kaal" },
      { local:"ਧੰਨਵਾਦ (Dhannvad)",    meaning:"Thank you",            phonetic:"Dhan-vaad" },
      { local:"ਕਿਵੇਂ ਹੋ? (Kiven ho?)", meaning:"How are you?",        phonetic:"Ki-ven ho?" },
      { local:"ਕਿੰਨਾ? (Kinna?)",      meaning:"How much?",            phonetic:"Kin-na?" },
      { local:"ਵਾਹਿਗੁਰੂ (Waheguru)",  meaning:"Praise God (common expression)", phonetic:"Wa-he-gu-ru" },
      { local:"ਹਾਂ / ਨਹੀਂ",          meaning:"Yes / No",            phonetic:"Haan / Na-hin" },
      { local:"ਕਿੱਥੇ? (Kithe?)",      meaning:"Where?",               phonetic:"Kit-the?" },
      { local:"ਬਹੁਤ ਵਧੀਆ (Bahut vadhia)", meaning:"Very good / Excellent", phonetic:"Ba-hut vad-hi-a" },
    ],
  },

  konkani: {
    language: "Konkani", script: "Konknni",
    phrases: [
      { local:"Dev borem korum",       meaning:"Thank you (God bless)", phonetic:"Dev bo-rem ko-rum" },
      { local:"Kitlem zhata?",         meaning:"How much is it?",       phonetic:"Kit-lem zha-ta?" },
      { local:"Bore assa (Borem assa)", meaning:"It's good",            phonetic:"Bo-re as-sa" },
      { local:"Maka samjona",          meaning:"I don't understand",    phonetic:"Ma-ka sam-jo-na" },
      { local:"Koshem assa?",          meaning:"How are you?",          phonetic:"Ko-shem as-sa?" },
      { local:"Vo / Naka",            meaning:"Yes / No",              phonetic:"Vo / Na-ka" },
      { local:"Xevott kainch na",      meaning:"No problem",           phonetic:"She-vot kainch na" },
      { local:"Koddi khavunk meltha?", meaning:"Can I get fish curry?", phonetic:"Ko-di kha-vunk mel-tha?" },
    ],
  },

  khasi: {
    language: "Khasi", script: "Khasi",
    phrases: [
      { local:"Khublei (ᱠᱷᱩᱵᱞᱮ)",    meaning:"Thank you",            phonetic:"Koo-blay" },
      { local:"Kumno phi long?",       meaning:"How are you?",         phonetic:"Koom-no fee long?" },
      { local:"Sngewbha",              meaning:"Please",               phonetic:"Sng-web-ha" },
      { local:"Phi long katno?",       meaning:"What is your name?",   phonetic:"Fee long kat-no?" },
      { local:"Kumno ban leit?",       meaning:"How to go?",           phonetic:"Koom-no ban lait?" },
      { local:"Nganam mynta",          meaning:"I am hungry",          phonetic:"Nga-nam min-ta" },
      { local:"Hep / Biang",          meaning:"Yes / No",             phonetic:"Hep / Bi-ang" },
      { local:"Ia phi",                meaning:"You're welcome",       phonetic:"Ee fee" },
    ],
  },

  assamese: {
    language: "Assamese", script: "অসমীয়া",
    phrases: [
      { local:"নমস্কাৰ (Nomoskar)",    meaning:"Hello",                phonetic:"No-mos-kar" },
      { local:"ধন্যবাদ (Dhonyobad)",  meaning:"Thank you",            phonetic:"Dhon-yo-bad" },
      { local:"আপুনি কেনে আছে?",      meaning:"How are you?",         phonetic:"Aa-pu-ni ke-ne a-se?" },
      { local:"কিমান দাম? (Kiman dam?)", meaning:"How much?",         phonetic:"Ki-man dam?" },
      { local:"মই নুবুজো",            meaning:"I don't understand",   phonetic:"Moi nu-bu-jo" },
      { local:"ক'ত? (Kot?)",          meaning:"Where?",               phonetic:"Ko-t?" },
      { local:"হয় / নহয়",            meaning:"Yes / No",            phonetic:"Hoy / No-hoy" },
      { local:"সহায় কৰক",            meaning:"Please help",          phonetic:"So-hai ko-rok" },
    ],
  },

  odia: {
    language: "Odia", script: "ଓଡ଼ିଆ",
    phrases: [
      { local:"ନମସ୍କାର (Namaskar)",   meaning:"Hello",                phonetic:"Na-mas-kar" },
      { local:"ଧନ୍ୟବାଦ (Dhanyabad)",  meaning:"Thank you",            phonetic:"Dhan-ya-bad" },
      { local:"ଆପଣ କେମିତି ଅଛନ୍ତି?",  meaning:"How are you?",         phonetic:"Aa-pan ke-mi-ti a-chan-ti?" },
      { local:"କେତେ ଦାମ? (Kete dam?)", meaning:"How much?",           phonetic:"Ke-te dam?" },
      { local:"ମୁଁ ବୁଝୁ ନାହିଁ",       meaning:"I don't understand",   phonetic:"Mun buj-hu na-hin" },
      { local:"କୁଆଡ଼େ? (Kuade?)",     meaning:"Where?",               phonetic:"Ku-a-de?" },
      { local:"ହଁ / ନାଁ",             meaning:"Yes / No",            phonetic:"Han / Naan" },
      { local:"ସାହାଯ୍ୟ କରନ୍ତୁ",      meaning:"Please help",          phonetic:"Sa-ha-jyo ka-ran-tu" },
    ],
  },

  kashmiri: {
    language: "Kashmiri / Urdu", script: "کشمیری",
    phrases: [
      { local:"آداب (Aadab)",          meaning:"Hello (respectful)",   phonetic:"Aa-daab" },
      { local:"شکریہ (Shukriya)",      meaning:"Thank you",            phonetic:"Shuk-ri-ya" },
      { local:"کیا حال ہے؟ (Kya haal hai?)", meaning:"How are you?", phonetic:"Kya haal hai?" },
      { local:"کتنا? (Kitna?)",        meaning:"How much?",            phonetic:"Kit-na?" },
      { local:"بہت خوبصورت",          meaning:"Very beautiful",        phonetic:"Ba-hut khu-b-su-rat" },
      { local:"ہاں / نہیں",           meaning:"Yes / No",             phonetic:"Haan / Na-hin" },
      { local:"آرام سے (Aaram se)",    meaning:"Take it easy / Slowly", phonetic:"Aa-ram se" },
      { local:"خوش آمدید (Khush Amdeed)", meaning:"Welcome",         phonetic:"Khush Am-deed" },
    ],
  },

  nepali: {
    language: "Nepali / Sikkimese", script: "नेपाली",
    phrases: [
      { local:"नमस्ते (Namaste)",       meaning:"Hello",                phonetic:"Na-mas-te" },
      { local:"धन्यवाद (Dhanyabad)",   meaning:"Thank you",            phonetic:"Dhan-ya-baad" },
      { local:"कस्तो छ? (Kasto chha?)", meaning:"How are you?",        phonetic:"Kas-to chha?" },
      { local:"कति? (Kati?)",          meaning:"How much?",            phonetic:"Ka-ti?" },
      { local:"राम्रो छ (Ramro chha)", meaning:"It's good / nice",    phonetic:"Ram-ro chha" },
      { local:"छ / छैन",               meaning:"Yes / No",            phonetic:"Chha / Chhain" },
      { local:"माफ गर्नुस्",           meaning:"Excuse me / Sorry",    phonetic:"Maaf gar-nu-hos" },
      { local:"मलाई थाहा छैन",         meaning:"I don't know",         phonetic:"Ma-lai tha-ha chhain" },
    ],
  },

  bhojpuri: {
    language: "Hindi / Bhojpuri", script: "हिन्दी",
    phrases: [
      { local:"प्रणाम (Pranaam)",      meaning:"Greetings (respectful)", phonetic:"Pra-naam" },
      { local:"धन्यवाद (Dhanyavad)",  meaning:"Thank you",            phonetic:"Dhan-ya-vaad" },
      { local:"का हाल-चाल बा?",       meaning:"How are you? (Bhojpuri)", phonetic:"Ka haal-chaal ba?" },
      { local:"कितना होई?",           meaning:"How much will it be?",  phonetic:"Kit-na ho-ee?" },
      { local:"बोधगया जाय के बा",    meaning:"I need to go to Bodhgaya", phonetic:"Bodh-ga-ya jay ke ba" },
      { local:"हँ / नाहीं",           meaning:"Yes / No",            phonetic:"Han / Na-hin" },
      { local:"बहुत नीमन",            meaning:"Very good (Bhojpuri)", phonetic:"Ba-hut nee-man" },
      { local:"कहाँ बा? (Kahan ba?)", meaning:"Where is it?",         phonetic:"Ka-han ba?" },
    ],
  },

  manipuri: {
    language: "Meitei / Hindi", script: "মণিপুরী",
    phrases: [
      { local:"হায় (Hay)",            meaning:"Hello / Hi",           phonetic:"Hay" },
      { local:"থাগৎচরি (Thagatcharee)", meaning:"Thank you",          phonetic:"Tha-gat-cha-ri" },
      { local:"নুংশিজরে? (Nungshijare?)", meaning:"How are you?",     phonetic:"Nung-shi-ja-re?" },
      { local:"কৎ চাউ? (Kat chau?)",  meaning:"How much?",            phonetic:"Kat chau?" },
      { local:"হৌদোক্লে (Houdokle)",  meaning:"OK / Alright",        phonetic:"Haw-dok-le" },
      { local:"হয়রে / নত্তে",        meaning:"Yes / No",            phonetic:"Hay-re / Not-te" },
      { local:"কদায়? (Kaday?)",       meaning:"Where?",               phonetic:"Ka-day?" },
      { local:"ফজবা লৌরিবা",          meaning:"Very beautiful",       phonetic:"Faj-ba lau-ri-ba" },
    ],
  },

  ladakhi: {
    language: "Ladakhi / Hindi", script: "Ladakhi",
    phrases: [
      { local:"Julley (জুলে)",         meaning:"Hello / Thank you / Goodbye", phonetic:"Ju-lay" },
      { local:"Khamzang yod-de?",      meaning:"How are you?",         phonetic:"Kham-zang yod-day?" },
      { local:"Thujeche",              meaning:"Thank you",            phonetic:"Thu-je-che" },
      { local:"Ringmo katse?",         meaning:"How much?",            phonetic:"Ring-mo kat-say?" },
      { local:"Nyinchung nyingje-po",  meaning:"Very beautiful",       phonetic:"Nyin-chung nying-je-po" },
      { local:"Yes-o / Mano",          meaning:"Yes / No",            phonetic:"Yeh-so / Ma-no" },
      { local:"Pangong la kamat duk?", meaning:"How far is Pangong?",  phonetic:"Pan-gong la ka-mat duk?" },
      { local:"Gongma-ri duk?",        meaning:"Is there snow?",       phonetic:"Gong-ma-ri duk?" },
    ],
  },

  // Generic Hindi fallback
  hindi_mp: {
    language: "Hindi", script: "हिन्दी",
    phrases: [
      { local:"नमस्ते (Namaste)",      meaning:"Hello",                phonetic:"Na-mas-tay" },
      { local:"धन्यवाद (Dhanyavad)",  meaning:"Thank you",            phonetic:"Dhan-ya-vaad" },
      { local:"क्या हाल है?",         meaning:"How are you?",         phonetic:"Kya haal hai?" },
      { local:"कितना? (Kitna?)",       meaning:"How much?",            phonetic:"Kit-na?" },
      { local:"कहाँ? (Kahan?)",        meaning:"Where?",               phonetic:"Ka-han?" },
      { local:"हाँ / नहीं",           meaning:"Yes / No",            phonetic:"Haan / Na-hin" },
      { local:"मदद करें (Madad karen)", meaning:"Please help",        phonetic:"Ma-dad ka-ren" },
      { local:"बहुत सुंदर",           meaning:"Very beautiful",        phonetic:"Ba-hut sun-dar" },
    ],
  },

  hindi_ne: {
    language: "Hindi / Tribal language", script: "हिन्दी",
    phrases: [
      { local:"नमस्ते (Namaste)",      meaning:"Hello",                phonetic:"Na-mas-tay" },
      { local:"धन्यवाद (Dhanyavad)",  meaning:"Thank you",            phonetic:"Dhan-ya-vaad" },
      { local:"क्या हाल है?",         meaning:"How are you?",         phonetic:"Kya haal hai?" },
      { local:"कितना? (Kitna?)",       meaning:"How much?",            phonetic:"Kit-na?" },
      { local:"हाँ / नहीं",           meaning:"Yes / No",            phonetic:"Haan / Na-hin" },
      { local:"मदद करें",             meaning:"Please help",          phonetic:"Ma-dad ka-ren" },
      { local:"बहुत सुंदर",           meaning:"Very beautiful",        phonetic:"Ba-hut sun-dar" },
      { local:"कहाँ है?",             meaning:"Where is it?",         phonetic:"Ka-han hai?" },
    ],
  },
};

/* Generic fallback */
const FALLBACK = {
  language: "Hindi (National)",
  script: "हिन्दी",
  phrases: [
    { local:"नमस्ते (Namaste)",       meaning:"Hello",                phonetic:"Na-mas-tay" },
    { local:"धन्यवाद (Dhanyavad)",   meaning:"Thank you",            phonetic:"Dhan-ya-vaad" },
    { local:"कितना? (Kitna?)",        meaning:"How much?",            phonetic:"Kit-na?" },
    { local:"कहाँ? (Kahan?)",         meaning:"Where?",               phonetic:"Ka-han?" },
    { local:"हाँ / नहीं",            meaning:"Yes / No",            phonetic:"Haan / Na-hin" },
    { local:"मदद करें (Madad karen)", meaning:"Please help",          phonetic:"Ma-dad ka-ren" },
    { local:"माफ़ करना (Maaf karna)", meaning:"Excuse me / Sorry",    phonetic:"Maaf kar-na" },
    { local:"बहुत अच्छा (Bahut acchha)", meaning:"Very good",        phonetic:"Ba-hut aach-ha" },
  ],
};

function detectLanguage(destination) {
  if (!destination) return FALLBACK;
  const lower = destination.toLowerCase().trim();
  // Check against all keywords
  for (const [keyword, langKey] of Object.entries(KEYWORD_TO_LANG)) {
    if (lower.includes(keyword)) {
      return LANG_DATA[langKey] || FALLBACK;
    }
  }
  return FALLBACK;
}

export default function LanguageTips({ destination }) {
  const [copied, setCopied] = useState(null);
  const data = detectLanguage(destination);

  const copyPhrase = (text) => {
    try { navigator.clipboard.writeText(text); } catch {}
    setCopied(text);
    setTimeout(() => setCopied(null), 2000);
  };

  return (
    <div className="card">
      <div className="card-header">
        <span className="card-title">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight:6 }}>
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
          </svg>
          Language Tips
        </span>
        <span className="pill blue" style={{ display:"flex", alignItems:"center", gap:5 }}>
          🇮🇳 {data.language}
        </span>
      </div>
      <div className="card-body">
        <div style={{ fontSize:12.5, color:"var(--text-2)", marginBottom:14, display:"flex", alignItems:"center", gap:8 }}>
          <span>Useful phrases for <strong style={{ color:"var(--text)" }}>{destination || "your destination"}</strong> in {data.language}</span>
          {data.script && <span style={{ fontFamily:"serif", fontSize:15, color:"var(--text-3)" }}>({data.script})</span>}
        </div>
        <div className="lang-grid">
          {data.phrases.map((p, i) => (
            <motion.div key={i} className="lang-phrase"
              initial={{ opacity:0, y:6 }} animate={{ opacity:1, y:0 }} transition={{ delay:i*0.05 }}
              onClick={() => copyPhrase(p.local)}
              style={{ cursor:"pointer", position:"relative", transition:"all 0.15s" }}
              whileHover={{ borderColor:"var(--blue-border)", background:"white" }}>
              <div className="lang-local">{p.local}</div>
              <div className="lang-meaning">{p.meaning}</div>
              <div className="lang-phonetic">{p.phonetic}</div>
              {copied === p.local && (
                <div style={{ position:"absolute", top:6, right:8, fontSize:10, color:"var(--green)", fontWeight:700, background:"var(--green-bg)", padding:"2px 6px", borderRadius:"var(--r-full)" }}>
                  Copied!
                </div>
              )}
            </motion.div>
          ))}
        </div>
        <div style={{ marginTop:12, padding:"10px 12px", background:"var(--blue-dim)", borderRadius:"var(--r-md)", border:"1.5px solid var(--blue-border)", fontSize:12, color:"var(--blue)" }}>
          💡 Click any phrase to copy it. Save offline — internet may be spotty in remote areas.
        </div>
      </div>
    </div>
  );
}
