
LOAD CSV WITH HEADERS FROM 'file:///routes.csv' AS row

MERGE (a:Airlines {airlines: 'Airlines' })
MERGE (pu:Punctuality {punctuality : toFloat(row.punctuality)})
MERGE (sq:Service_Quality {service_quality : toFloat(row.serviceQuality)})
MERGE (hp:Handling_passengers {handling_passenger : toFloat(row.handlingPassengers)})
MERGE (ah:AirHelp {airhelp : toFloat(row.airHelpScore)})
MERGE (dc:DestinationCity {destinationCity: row.destinationCity})
MERGE (sc:SourceCity {SourceCity: row.sourceCity})
MERGE (sa:SourceAirport {sourceAirport: row.sourceAirport, sourceAirportName: row.sourceAirportName,DEPARTURE_DELAY:row.DEPARTURE_DELAY })
MERGE (da:DestinationAirport {destinationAirport: row.destinationAirport, destinationAirportName: row.destinationAirportName,ARRIVAL_DELAY:row.ARRIVAL_DELAY})
MERGE (al:Airline {airline: row.airline, airlineName: row.airlineName, dep_date: date({year: toInteger(row.departureYear), month: toInteger(row.departureMonth), day: toInteger(row.departureDay)}), arr_date: date({year: toInteger(row.arrivalYear), month: toInteger(row.arrivalMonth), day: toInteger(row.arrivalDay)}) ,overall_rating:row.overall_rating,
          seat_comfort_rating:row.seat_comfort_rating,cabin_staff_rating:row.cabin_staff_rating,
          food_beverages_rating:row.food_beverages_rating,inflight_entertainment_rating:row.inflight_entertainment_rating,
          value_money_rating:row.value_money_rating})
CREATE (a) - [:PUNCTUALITY] -> (pu)
CREATE (a) - [:SERVICE_QUALITY] -> (sq)
CREATE (a) - [:HANDLING_PASSENGERS] -> (hp)
CREATE (a) - [:AIRHELP] -> (ah)
CREATE (pu) - [:AIRLINE] -> (al)
CREATE (sq) - [:AIRLINE] -> (al)
CREATE (hp) - [:AIRLINE] -> (al)
CREATE (ah) - [:AIRLINE] -> (al)
CREATE (a) - [:START_FROM] -> (dc)
CREATE (sc) <- [:SOURCE_CITY] - (sa) - [:HAS_FLIGHT{AIR_TIME:row.AIR_TIME,APPID:row.APPID}] -> (al) - [:TO_DESTINATION] -> (da) - [:DEST_CITY] -> (dc)
