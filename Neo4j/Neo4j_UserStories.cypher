
//User Story 01: 
MATCH (n:Handling_passengers) RETURN n LIMIT 25


//User Story 02:
MATCH (n:Service_Quality) RETURN n LIMIT 25


//User Story 03:
MATCH (sq:Service_Quality), (pu:Punctuality), (hp:Handling_passengers), (ah:AirHelp)

WITH max(sq.service_quality) AS sqmax , max(pu.punctuality) AS pumax, max(hp.handling_passenger) AS hpmax, max(ah.airhelp) AS ahmax

MATCH (a:Airlines) - [:SERVICE_QUALITY] -> (sq:Service_Quality), (a:Airlines) - [:PUNCTUALITY] -> (pu:Punctuality), 
(a:Airlines) - [:HANDLING_PASSENGERS] -> (hp:Handling_passengers), (a:Airlines) - [:AIRHELP] -> (ah:AirHelp)

WHERE sq.service_quality = sqmax and pu.punctuality = pumax and hp.handling_passenger = hpmax and ah.airhelp = ahmax
RETURN a, sq, pu, ah, hp 


//User Story 04:
WITH date("2020-09-15") AS first

MATCH (al:Airline), (sa:SourceAirport), (da:DestinationAirport), (sc:SourceCity), (dc:DestinationCity)

MATCH (sc) <- [:SOURCE_CITY] - (sa) - [:HAS_FLIGHT] -> (al)
MATCH (al) - [:TO_DESTINATION] -> (da) - [:DEST_CITY] -> (dc) 

WHERE ( al.dep_date = first and sa.sourceAirport = 'AAL' and da.destinationAirport = 'AMS' and al.arr_date = first) OR (  al.dep_date = first + duration('P2D') and sa.sourceAirport = 'AMS' and da.destinationAirport = 'OSL' and al.arr_date = first + duration('P2D')) OR  (  al.dep_date = first + duration('P4D') and sa.sourceAirport = 'OSL' and da.destinationAirport = 'AAL' and al.arr_date = first + duration('P4D')) 
RETURN al, sa, da, sc, dc


//User Story 05:
MERGE (c:Community {community: row.Community}) 
MERGE (com:Total_complaints {complaints: toInteger(row.Total_complaints), year: toInteger(row.Noise_complaint_year), month: toInteger(row.Noise_complaint_month)}) 
MERGE (cal:Total_number_of_callers {callers: toInteger(row.Total_number_of_callers), year: toInteger(row.Noise_complaint_year), month: toInteger(row.Noise_complaint_month)}) 
MERGE (c) - [:COMPLAINTS] -> (com)
MERGE (c) - [:CALLERS] -> (cal)

MATCH (c:Community) - [r:COMPLAINTS] -> (com:Total_complaints) 
RETURN c, SUM(com.complaints) AS Total  
ORDER BY Total DESC 
LIMIT 3
