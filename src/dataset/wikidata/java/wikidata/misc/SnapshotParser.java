package wikidata.misc;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;

import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashSet;
import java.util.Set;

public class SnapshotParser {
    JsonFactory jfactory = new JsonFactory();
    SimpleDateFormat dateFormat = new SimpleDateFormat("+yyyy-MM-dd'T'hh:mm:ss'Z'");

    public void addTriple(String subjectQid, String propertyId, String objectQid, Set<KGTripleInt> readTriples) {
        assert objectQid.startsWith("Q");
        assert propertyId.startsWith("P");
        assert subjectQid.startsWith("Q");
        int objectQidQualifierInt =
                Integer.parseInt(objectQid.substring(1));
        int propertyIdQualifierInt =
                Integer.parseInt(propertyId.substring(1));
        int subjectQidQualifierInt =
                Integer.parseInt(subjectQid.substring(1));
        KGTripleInt kgTripleInt = new KGTripleInt(subjectQidQualifierInt,
                propertyIdQualifierInt, objectQidQualifierInt);
        readTriples.add(kgTripleInt);
    }

    public void addTripleStr(String subjectQid, String propertyId, String objectQid, Set<KGTriple> readTriples,
                             boolean isQualifier, Date timestamp) {
        assert objectQid.startsWith("Q");
        assert propertyId.startsWith("P");
        assert subjectQid.startsWith("Q");
        KGTriple kgTriple = new KGTriple(subjectQid, propertyId, objectQid, isQualifier, timestamp);
        readTriples.add(kgTriple);
    }

    public void addDoubleExtractor(String subjectQid, String propertyId,
                                   String objectValue, String objectType,
                                   Set<KGDouble> readDoubles,
                                   TempQualifierV2 doubleTempQualifier) {
        assert subjectQid.startsWith("Q");
        assert propertyId.startsWith("P");
        KGDouble kgDouble = new KGDouble(propertyId, objectValue, objectType);
        kgDouble.setTempQualifier(doubleTempQualifier);
        readDoubles.add(kgDouble);
    }

    private String parseDateTime(String dateToParseSB) {
        int year = -99999;
        boolean yearAssigned = false;
        int month = -99999;
        boolean monthAssigned = false;
        int day = -99999;
        boolean dayAssigned = false;

        int hour = -99999;
        boolean hourAssigned = false;
        int minute = -99999;
        boolean minuteAssigned = false;
        int second = -99999;
        boolean secondAssigned = false;
        byte currentPart = 0;
        StringBuilder currentElement = new StringBuilder();

        for (int i = 0; i < dateToParseSB.length(); i++) {
            char currentChar = dateToParseSB.charAt(i);
            if (currentChar == '-' && i > 0) {
                // here process the month or year
                if (currentPart == 0) {
                    year = Integer.parseInt(currentElement.toString());
                    yearAssigned = true;
                } else if (currentPart == 1) {
                    month = Integer.parseInt(currentElement.toString());
                    monthAssigned = true;
                }
                currentElement = new StringBuilder();
                currentPart += 1;
            } else if (currentChar == 'T') {
                // here process the day
                day = Integer.parseInt(currentElement.toString());
                dayAssigned = true;
                assert currentPart == 2;
                assert currentElement.length() == 2;
                currentPart += 1;
                currentElement = new StringBuilder();
            } else if (currentChar == ':') {
                // here process the hour or minute
                if (currentPart == 3) {
                    hour = Integer.parseInt(currentElement.toString());
                    hourAssigned = true;
                } else if (currentPart == 4) {
                    minute = Integer.parseInt(currentElement.toString());
                    minuteAssigned = true;
                }

                assert currentElement.length() == 2;
                currentPart += 1;
                currentElement = new StringBuilder();
            } else if (currentChar == 'Z') {
                // TODO: here process the second
                second = Integer.parseInt(currentElement.toString());
                secondAssigned = true;
                assert currentElement.length() == 2;
                assert currentPart == 5;
                assert i == dateToParseSB.length() - 1;
                currentPart += 1;
            } else {
                currentElement.append(currentChar);
            }
        }
        assert yearAssigned &&
                monthAssigned &&
                dayAssigned &&
                hourAssigned &&
                minuteAssigned &&
                secondAssigned;

        StringBuilder myTimestamp = new StringBuilder("Y" + year);
        if (month > 0) {
            myTimestamp.append("MM").append(month);
        }
        if (day > 0) {
            myTimestamp.append("D").append(day);
        }
        if (hour > 0) {
            myTimestamp.append("H").append(hour);
        }
        if (minute > 0) {
            myTimestamp.append("M").append(minute);
        }
        if (second > 0) {
            myTimestamp.append("S").append(second);
        }

        return myTimestamp.toString();
    }

    public void parseSnapshotExtractor(Set<KGDouble> readDoubles, JSONSnapshot currSnapshot) throws IOException, ParseException {
        // parses the snapshot to create the actual dataset
        JsonParser jParser = jfactory.createParser(currSnapshot.getText());

        int depth = 0;
        JsonToken nextToken = jParser.nextToken();
        boolean insideClaims = false;
        boolean insideMainSnak = false;
        boolean insideDataValue = false;
        boolean insideValue = false;
        boolean insideQualifiers = false;
//        boolean recognizedQualifier = false;
        boolean insideQualifierDataValue = false;
        boolean insideQualifierValue = false;
        boolean insideTemporalQualifier = false;

        String propertyId = "";
        String subjectQid = "";
        StringBuilder objectValue = new StringBuilder();
        String objectType = "";
        String objectUnit = "";
        String propertyQualifierId = "";
        String currQualifierEntityType = "";
        String currEntityType = "";
        String currTemporalPrecision = "";
        StringBuilder currTempQualifierValue = new StringBuilder();
        boolean doAdd = true;
        Set<KGDouble> readCurrDoubles = new HashSet<KGDouble>();
        // TODO: define for list of qualifiers
//        ArrayList<TempQualifier> doubleTempQualifiers = new ArrayList<TempQualifier>();
//        TempQualifier doubleTempQualifier = new TempQualifier();
        TempQualifierV2 doubleTempQualifier = new TempQualifierV2();
        while (nextToken != null) {
            if (nextToken == JsonToken.START_OBJECT) {
                depth += 1;
            }
            if (nextToken == JsonToken.START_ARRAY) {
                depth += 1;
            }
            if (depth == 1 && "claims".equals(jParser.getCurrentName()) && nextToken == JsonToken.FIELD_NAME) {
                insideClaims = true;
            }
            if (depth == 4 && insideClaims && nextToken == JsonToken.FIELD_NAME) {
                if ("mainsnak".equals(jParser.getCurrentName())) {
                    insideMainSnak = true;
                }
                if ("qualifiers".equals(jParser.getCurrentName())) {
                    doubleTempQualifier = new TempQualifierV2();
                    insideQualifiers = true;
                    // TODO: here initialize doubleQualifiers
                }
            }
            if (depth == 5 && insideMainSnak && nextToken == JsonToken.FIELD_NAME) {
                if ("datavalue".equals(jParser.getCurrentName())) {
                    insideDataValue = true;
                    doAdd = false;
                }
            }
            if (depth == 6 && insideDataValue && nextToken == JsonToken.FIELD_NAME) {
                insideValue = true;
            }
            if (depth == 6 && insideDataValue && nextToken == JsonToken.VALUE_STRING) {
                if ("value".equals(jParser.getCurrentName())) {
                    currEntityType = "";
                    // tries to obtain the value , null will be returned if value structure inside
                    String strVal = jParser.getValueAsString();

                    objectValue = new StringBuilder();
                    if (strVal != null & (!strVal.equals("value"))) {
                        objectValue.append(strVal);
                        doAdd = true;
                    } else {
                        System.out.println("objectValue after being assigned: " + strVal + " in null or not valid");
                    }

                } else if ("type".equals(jParser.getCurrentName())) {
                    objectType = jParser.getValueAsString();
                }
            }

            //TODO: return this line back with recognizedQualifier, will be faster!!
//            if (depth == 7 && recognizedQualifier && insideQualifiers && nextToken == JsonToken.FIELD_NAME) {
            if (depth == 7 && insideQualifiers && nextToken == JsonToken.FIELD_NAME) {
                if ("datavalue".equals(jParser.getCurrentName())) {
                    insideQualifierDataValue = true;
                }
            }
            //TODO: return this line back with recognizedQualifier, will be faster!!
//            if (depth == 8 && recognizedQualifier && insideQualifierDataValue && nextToken == JsonToken.FIELD_NAME) {
            if (depth == 8 && insideQualifierDataValue && nextToken == JsonToken.FIELD_NAME) {
                insideQualifierValue = true;
            }
            if (depth == 9 &&
//                    recognizedQualifier &&
                    insideQualifierValue &&
                    insideQualifierDataValue &&
                    (nextToken == JsonToken.VALUE_STRING || nextToken == JsonToken.VALUE_NUMBER_INT)) {

                if (jParser.getCurrentName().equals("time")) {
                    String currQualifierValue = jParser.getValueAsString();
                    try {
                        insideTemporalQualifier = true;
                        String qualifierTime = parseDateTime(currQualifierValue);
                        currTempQualifierValue.append(propertyQualifierId)
                                .append(":")
                                .append(qualifierTime);
//                        doubleTempQualifier.addTemporalQualifier(
//                                propertyQualifierId + ":" + qualifierTime);
//                        switch (propertyQualifierId) {
//                            case "P570" -> doubleTempQualifier.addDateOfDeath(qualifierTime);
//                            case "P574" -> doubleTempQualifier.addPublicationTaxonYear(qualifierTime);
//                            case "P577" -> doubleTempQualifier.addPublicationDate(qualifierTime);
//                            case "P585" -> doubleTempQualifier.addPointInTime(qualifierTime);
//                            case "P580" -> doubleTempQualifier.addStartTime(qualifierTime);
//                            case "P582" -> doubleTempQualifier.addEndTime(qualifierTime);
//                            case "P1326" -> doubleTempQualifier.addLatestDate(qualifierTime);
//                            case "P1319" -> doubleTempQualifier.addEarliestDate(qualifierTime);
//                            //archive date
//                            case "P2960" -> doubleTempQualifier.addArchiveDate(qualifierTime);
//                            // retrieved
//                            case "P813" -> doubleTempQualifier.addRetrievedDate(qualifierTime);
//                            // inception
//                            case "P571" -> doubleTempQualifier.addInceptionDate(qualifierTime);
//                            // burial or cremation date
//                            case "P4602" -> doubleTempQualifier.addBurialOrCremationDate(qualifierTime);
//                            // date of birth
//                            case "P569" -> doubleTempQualifier.addBirthDate(qualifierTime);
//                            // start of covered period
//                            case "P7103" -> doubleTempQualifier.addStartOfCoveredPeriodDate(qualifierTime);
//                            // end of covered period
//                            case "P7104" -> doubleTempQualifier.addEndOfCoveredPeriodDate(qualifierTime);
//                            default -> System.out.println("Not recognized propertyQualifierId: " +
//                                    propertyQualifierId);
//                        }
//                        if (subjectQid.equals("Q30098005")) {
//                            System.out.println("detail on Q30098005 when parsing the following date: " +
//                                    currQualifierValue + " more concretely: " +
//                                    " subject: " + subjectQid +
//                                    " property: " + propertyId +
//                                    " Value: " + objectValue +
//                                    " Qualifier: " + propertyQualifierId +
//                                    " Value: " + currQualifierValue);
//                        }
//                        switch (propertyQualifierId) {
//                            case "P585" -> doubleTempQualifier.addPointInTime(qualifierTime.getTime());
//                            case "P580" -> doubleTempQualifier.addStartTime(qualifierTime.getTime());
//                            case "P582" -> doubleTempQualifier.addEndTime(qualifierTime.getTime());
//                            default -> System.out.println("Not recognized propertyQualifierId: " +
//                                    propertyQualifierId);
//                        }
                    } catch (Exception ex) {
                        System.out.println("exception when parsing the following date: " +
                                currQualifierValue + " more concretely: " +
                                " subject: " + subjectQid +
                                " property: " + propertyId +
                                " Value: " + objectValue +
                                " Qualifier: " + propertyQualifierId +
                                " Value: " + currQualifierValue);
                        ex.printStackTrace();
                    }
                } else if (jParser.getCurrentName().equals("precision")) {
                    currTemporalPrecision = jParser.getValueAsString();
//                    System.out.println("precision to " + currTemporalPrecision);
                }
            }
//            else if (depth == 9 && insideQualifierValue &&
//                    insideQualifierDataValue && nextToken == JsonToken.VALUE_STRING) {
//                if (jParser.getCurrentName().equals("time")) {
//                    System.out.println("Not recognized propertyQualifierId of time: " +
//                            propertyQualifierId);
//                }
//            }
            if (depth == 1 && nextToken == JsonToken.VALUE_STRING) {
                if ("id".equals(jParser.getCurrentName())) {
                    subjectQid = jParser.getValueAsString();
//                    if (subjectQid.equals("Q30098005")) {
//                        System.out.println(currSnapshot.getText());
//                    }
                }
            }
            if (depth == 2 && insideClaims && nextToken == JsonToken.FIELD_NAME) {
                propertyId = jParser.getCurrentName();
            }
            if (depth == 5 && insideQualifiers && nextToken == JsonToken.FIELD_NAME) {
                propertyQualifierId = jParser.getCurrentName();
//                if (
//                        propertyQualifierId.equals("P574") ||
//                        propertyQualifierId.equals("P577") ||
//                        propertyQualifierId.equals("P570") ||
//                        propertyQualifierId.equals("P585") ||
//                        propertyQualifierId.equals("P580") ||
//                        propertyQualifierId.equals("P582") ||
//                        propertyQualifierId.equals("P1319") ||
//                        propertyQualifierId.equals("P1326") ||
//                        propertyQualifierId.equals("P2960") ||
//                        propertyQualifierId.equals("P813") ||
//                        propertyQualifierId.equals("P571") ||
//                        propertyQualifierId.equals("P4602") ||
//                        propertyQualifierId.equals("P569") ||
//                        propertyQualifierId.equals("P7103") ||
//                        propertyQualifierId.equals("P7104")
//                ) {
//                    recognizedQualifier = true;
//                }
            }
            if (depth == 7 && insideValue &&
                    ((nextToken == JsonToken.VALUE_STRING) ||
                            (nextToken == JsonToken.VALUE_NUMBER_INT)
                    )

            ) {
                if ("entity-type".equals(jParser.getCurrentName())) {
                    currEntityType = jParser.getValueAsString();
                }

                if ("numeric-id".equals(jParser.getCurrentName())) {
                    if (currEntityType.equals("item") && objectValue.isEmpty()) {
                        doAdd = true;
                        objectValue.append("Q");
                        objectValue.append(jParser.getValueAsString());
                        objectType = "entity";
                        if (objectValue.toString().equals("Q13442814") ||
                                objectValue.toString().equals("Q7318358")) {
                            doAdd = false;
                        }
                    }
                } else if ("id".equals(jParser.getCurrentName())) {
                    if (currEntityType.equals("item") && objectValue.isEmpty()) {
                        doAdd = true;
                        objectValue.append(jParser.getValueAsString());
                        objectType = "entity";
                        if (objectValue.toString().equals("Q13442814") ||
                                objectValue.toString().equals("Q7318358")) {
                            doAdd = false;
                        }
                    }
                } else if ("time".equals(jParser.getCurrentName())) {
                    objectValue.append(jParser.getValueAsString());
                    objectType = "time";
                    doAdd = true;
                } else if ("latitude".equals(jParser.getCurrentName())) {
                    objectValue
                            .append("lat")
                            .append(jParser.getValueAsString())
                            .append(":");
                    objectType = "globecoordinate";
                    doAdd = true;
                } else if ("longitude".equals(jParser.getCurrentName())) {
                    objectValue
                            .append("lon")
                            .append(jParser.getValueAsString())
                            .append(":");
                    objectType = "globecoordinate";
                    doAdd = true;
                } else if ("amount".equals(jParser.getCurrentName())) {
                    objectValue.append(jParser.getValueAsString());
                    objectType = "amount";
                    doAdd = true;
                }
            }
            nextToken = jParser.nextToken();
            if (nextToken == JsonToken.END_OBJECT || nextToken == JsonToken.END_ARRAY) {
                depth -= 1;
                if (depth == 1 && insideClaims) {
                    insideClaims = false;
                }
                if (depth == 3 && insideClaims) {
                    if (doAdd) {
                        if (propertyId.equals("")) {
                            System.out.println("ERROR when processing subjectQid: " + subjectQid +
                                    " propertyId: " + propertyId +
                                    " objectValue: " + objectValue +
                                    " currSnapshot: " + currSnapshot.getText());
                        }
//                        else {
//                            System.out.println("everything nice, adding " +
//                                    "<" + subjectQid + "> <" +
//                                    "<" + propertyId + "> <" +
//                                    "<" + objectValue.toString() + "> <" +
//                                    objectType+ ">");
//                        }
                        addDoubleExtractor(subjectQid, propertyId, objectValue.toString(),
                                objectType, readCurrDoubles, doubleTempQualifier);
                        doubleTempQualifier = new TempQualifierV2();
                        objectType = "";
                    }
                    objectValue = new StringBuilder();
                    doAdd = false;
                }
                if (depth == 4 && insideMainSnak) {
                    insideMainSnak = false;
                }
                if (depth == 4 && insideQualifiers) {
                    insideQualifiers = false;
                }
//                if (depth == 5 && insideQualifiers && recognizedQualifier) {
//                    // end of a qualifier block, next one can possibly start
//                    recognizedQualifier = false;
//                }

                if (depth == 5 && insideDataValue) {
                    insideDataValue = false;
                }
                if (depth == 6 && insideValue) {
                    insideValue = false;
                }
                if (depth == 7 && insideQualifierDataValue) {
                    insideQualifierDataValue = false;
                    if (insideTemporalQualifier) {
                        currTempQualifierValue.append(":").append(currTemporalPrecision);
                        doubleTempQualifier.addTemporalQualifier(currTempQualifierValue.toString());
                        currTempQualifierValue = new StringBuilder();
                        insideTemporalQualifier = false;
                        currTemporalPrecision = "";
                    }
                }
                if (depth == 8 && insideQualifierValue) {
                    insideQualifierValue = false;
                }
            }
            if (nextToken == null && depth > 0) {
                System.out.println("something weird is going on with this json: " + currSnapshot.getText());
            }
        }
        readDoubles.addAll(readCurrDoubles);
        jParser.close();
    }
}
