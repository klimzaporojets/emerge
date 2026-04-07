package wikidata.misc;

import com.google.common.collect.Sets;
import com.opencsv.CSVWriter;
import org.apache.commons.compress.archivers.sevenz.SevenZArchiveEntry;
import org.apache.commons.compress.archivers.sevenz.SevenZFile;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;

import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamReader;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicLong;

public class ReaderJSONStream implements Runnable {
    private String pathToInputFile;
    private final BlockingQueue<JSONSnapshot> linkedBlockingQueue;
    //     private static AtomicInteger counter;
    private AtomicLong nrProcessedEntities;

    private Long startingTime;
    // 2012-11-26T06:44:31Z
    SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'");


    //    Set<KGTripleInt> readTriples = new HashSet<KGTripleInt>();
    Set<KGDouble> readDoubles = new HashSet<KGDouble>();

    SnapshotParser snapshotParser = new SnapshotParser();

    //    String outputPathTriples = "data/wikidata_processed/raw_triples_2/";
    String outputPathTriples;
    String outputPathEntities;
    FileFormat fileFormat;

    public enum FileFormat {
        SEVEN_ZIP,
        BZ2
    }

    public ReaderJSONStream(String pathToInputFile, BlockingQueue<JSONSnapshot> linkedBlockingQueue,
                            AtomicLong nrProcessedEntities,
                            String outputPathTriples,
                            String outputPathEntities,
                            FileFormat fileFormat) {
//        this.pathToInput7zHistoryFile = pathToInput7zHistoryFile;
        this.pathToInputFile = pathToInputFile;

        this.linkedBlockingQueue = linkedBlockingQueue;
        this.nrProcessedEntities = nrProcessedEntities;
        this.startingTime = System.currentTimeMillis();
        this.outputPathTriples = outputPathTriples;
        this.outputPathEntities = outputPathEntities;
        this.fileFormat = fileFormat;
    }

    public void processStream(InputStream inputStream, File outputFile,
                              File outputFileEntities,
                              XMLInputFactory xmlInputFactory) throws IOException, XMLStreamException, ParseException {
        List<String[]> csvData = new ArrayList<>();
        List<String[]> csvEntityData = new ArrayList<>();


//        CSVWriter csvWriter = new CSVWriter(new FileWriter(outputFile));
        CSVWriter csvWriterTriples = new CSVWriter(new FileWriter(outputFile),
                '\t', CSVWriter.NO_QUOTE_CHARACTER, '"', "\n");
        CSVWriter csvWriterEntities = new CSVWriter(new FileWriter(outputFileEntities),
                '\t', CSVWriter.NO_QUOTE_CHARACTER, '"', "\n");
        XMLStreamReader xmlStreamReader = xmlInputFactory.createXMLStreamReader(inputStream, "UTF-8");
        StringBuilder contentString = new StringBuilder();
        String currElement = "";
        String fieldTitleQid = "";
        String fieldText = "";
        String fieldNs = "";
        String fieldComment = "";
        Date fieldTimestamp = null;
        boolean processThisField = false;
        boolean processThisPage = true;
        int nrEntityRevisions = 0;
//        long revisionSize = 0;
//        HashMap<String, HashMap<KGDoubleInt, HashMap<String, ArrayList<Object>>>>
        HashMap<String, HashMap<KGDouble, ArrayList<long[]>>> triplesToHistory =
                new HashMap<>();

        HashMap<String, ArrayList<KGRevisionDouble>> subjectToRevisions =
                new HashMap<>();

        HashMap<String, HashMap<KGDouble, TempQualifierV2>> triplesToTempQualifiers =
                new HashMap<>();

        while (xmlStreamReader.hasNext()) {
            int xmlEvent = xmlStreamReader.next();
            //Process start element.
            if (xmlEvent == XMLStreamConstants.START_ELEMENT) {
//                            System.out.println("Start Element: " + xmlStreamReader.getLocalName());
                currElement = xmlStreamReader.getLocalName();
                switch (currElement) {
                    case "page" -> {
                        fieldTitleQid = "";
                    }
                    case "revision" -> {
                        fieldText = "";
                        fieldComment = "";
                    }
                    case "ns" -> processThisField = true;
                    case "comment" -> processThisField = true;
//                                case "comment" -> processThisField = false;
                    case "title" -> processThisField = true;
//                                case "text" -> processThisField = false;
                    case "text" -> processThisField = true;
                    case "timestamp" -> processThisField = true;
                    default -> processThisField = false;
                }
                if (processThisField) {
                    contentString = new StringBuilder();
                }
            }
            if ((processThisPage) && (processThisField) && (xmlEvent == XMLStreamConstants.CHARACTERS)) {
                contentString.append(xmlStreamReader.getText());
            }
            //Process end element.
            if (xmlEvent == XMLStreamConstants.END_ELEMENT) {
//                            System.out.println("currElement is " + currElement);
                currElement = xmlStreamReader.getLocalName();
                if (processThisPage && currElement.equals("text")) {
                    fieldText = contentString.toString();
                } else if (currElement.equals("ns")) {
                    fieldNs = contentString.toString();
                    if (!(fieldNs.equals("0"))) {
                        processThisPage = false;
                    } else {
                        processThisPage = true;
                    }
                } else if (processThisPage && currElement.equals("timestamp")) {
                    String toParse = contentString.toString();
                    fieldTimestamp = dateFormat.parse(toParse);
//                    System.out.println("ABOUT TO PARSE THE FOLLOWING INTO fieldTimestamp "
//                            + toParse + " into " + fieldTimestamp.getTime());
                } else if (processThisPage && currElement.equals("comment")) {
                    fieldComment = contentString.toString();
                } else if (processThisPage && currElement.equals("title")) {
                    fieldTitleQid = contentString.toString();
                } else if (currElement.equals("page")) {
                    if (processThisPage) {
                        Set<KGDouble> activeDoubles = new HashSet<KGDouble>();
                        // 10.08.2024 - this sort is necessary, since the wikidata revisions are not always in
                        // chronological order
                        subjectToRevisions.get(fieldTitleQid)
                                .sort(Comparator.comparingLong(KGRevisionDouble::getTimestamp));

                        // TODO: BEGIN delete the part below of Q6881113, only for debugging purposes
//                        if (fieldTitleQid.equals("Q6881113")) {
//                            System.out.println("Q6881113 was detected!!");
//                            for (KGRevisionDouble kgRevisionDouble : subjectToRevisions.get(fieldTitleQid)) {
//                                System.out.println("=======================================================");
//                                System.out.println("Q6881113 timestamp: " + kgRevisionDouble.getTimestamp());
//                                for (KGDouble currDouble : kgRevisionDouble.getKgDouble()) {
//                                    System.out.println("Q6881113 timestamp: " + kgRevisionDouble.getTimestamp() +
//                                            "\t" + currDouble.getPropertyId() + "\t" + currDouble.getObjectQid() +
//                                            "\t" + currDouble.getType());
//                                }
//                            }
//                        }

                        // TODO: END delete the part below of Q6881113, only for debugging purposes

                        triplesToHistory.putIfAbsent(fieldTitleQid, new HashMap<>());
                        nrEntityRevisions = 0;
                        for (KGRevisionDouble kgRevisionDouble : subjectToRevisions.get(fieldTitleQid)) {
                            nrEntityRevisions += 1;
                            Set<KGDouble> currDoubles = kgRevisionDouble.getKgDouble();
//                            if (currDoubles.isEmpty()){
//                                System.out.println("ERROR_currDoubles_empty for qid " + fieldTitleQid +
//                                        " and timestamp: " + kgRevisionDouble.getTimestamp());
//                            }
                            Set<KGDouble> deletedDoubles = Sets.difference(activeDoubles, currDoubles);
                            Set<KGDouble> addedDoubles = Sets.difference(currDoubles, activeDoubles);
                            if (nrEntityRevisions == 1) {
                                csvEntityData.add(new String[]{
                                        fieldTitleQid,
                                        String.valueOf(kgRevisionDouble.getTimestamp())
                                });
                                if (this.nrProcessedEntities.get() % 1000 == 0) {
                                    csvWriterEntities.writeAll(csvEntityData);
                                    csvEntityData = new ArrayList<>();
                                }
                            }

                            // adds (and replaces) temporal qualifiers for all the triples
                            for (KGDouble currDouble : currDoubles) {
                                triplesToTempQualifiers.get(fieldTitleQid)
                                        .put(currDouble, currDouble.getTempQualifier());
                            }


                            for (KGDouble currDeletedDouble : deletedDoubles) {
                                triplesToHistory.get(fieldTitleQid)
                                        .putIfAbsent(currDeletedDouble, new ArrayList<long[]>());
                                triplesToHistory.get(fieldTitleQid).get(currDeletedDouble)
                                        .add(new long[]{kgRevisionDouble.getTimestamp(), 0});
                            }

                            for (KGDouble currAddedDouble : addedDoubles) {
                                triplesToHistory.get(fieldTitleQid)
                                        .putIfAbsent(currAddedDouble, new ArrayList<long[]>());
                                triplesToHistory.get(fieldTitleQid).get(currAddedDouble)
                                        .add(new long[]{kgRevisionDouble.getTimestamp(), 1});
                            }
                            activeDoubles = currDoubles;
                        }
                        // end the code to sort the revisions by date

                        long currNr = this.nrProcessedEntities.getAndIncrement();
                        if (currNr % 10 == 0) {
                            csvData = new ArrayList<String[]>();

                            for (Map.Entry<String, HashMap<KGDouble, ArrayList<long[]>>>
                                    kgEntityEntry : triplesToHistory.entrySet()) {
                                HashMap<KGDouble, ArrayList<long[]>> currDoubleEntries =
                                        kgEntityEntry.getValue();
                                String tripleSubject = kgEntityEntry.getKey();
                                for (Map.Entry<KGDouble, ArrayList<long[]>>
                                        kgDoubleEntry : currDoubleEntries.entrySet()) {
                                    ArrayList<long[]> kgDoubleActions = kgDoubleEntry.getValue();
                                    KGDouble kgDoubleInt = kgDoubleEntry.getKey();
                                    String tripleRelation = kgDoubleInt.getPropertyId();
                                    String tripleObject = kgDoubleInt.getObjectQid();
                                    String type = kgDoubleInt.getType();
                                    StringBuilder stringTimestamps = new StringBuilder();
                                    for (long[] currAction : kgDoubleActions) {
                                        stringTimestamps.append(currAction[0])
                                                .append(":")
                                                .append((currAction[1] == 0) ? "D" : "A")
                                                .append(",");
                                    }
                                    TempQualifierV2 tempQualifier =
                                            triplesToTempQualifiers.get(tripleSubject).get(kgDoubleInt);

                                    for (String currTimestamp : tempQualifier.getTemporalQualifiers()) {
                                        stringTimestamps.append(currTimestamp)
                                                .append(",");
                                    }

                                    csvData.add(new String[]{
                                            tripleSubject,
                                            tripleRelation,
                                            tripleObject,
                                            type,
                                            stringTimestamps.toString()
                                    });
                                }
                            }

                            csvWriterTriples.writeAll(csvData);
                            triplesToHistory = new HashMap<>();
                            // 08.08.2024 - added initialization of triplesToTempQualifiers to clean the memory
                            triplesToTempQualifiers = new HashMap<>();
                            subjectToRevisions = new HashMap<>();
                        }
                    }
                    processThisPage = true;
                } else if (processThisPage && currElement.equals("revision")) {

                    // TODO: BEGIN delete this only for debugging purposes
//                    if (fieldTitleQid.equals("Q6881113")) {
//                        if (fieldTimestamp.getTime() == 1676127366000L) {
//                            System.out.println("Q6881113 - timestamp: " + fieldTimestamp.getTime() +
//                                    " Content: " + fieldText);
//                        }
//                    }
                    // TODO: END delete this only for debugging purposes

                    if (!fieldText.trim().isEmpty()) {
                        JSONSnapshot jsonSnapshot = new JSONSnapshot(fieldTitleQid, fieldText, fieldComment, fieldTimestamp);
//                        System.out.println("fieldText: " + fieldText);
//                    if (fieldTitleQid.equals("Q6881113")) {
//                        if (fieldTimestamp.getTime() == 1676127366000L) {
//                            System.out.println("Q6881113 - timestamp: " + fieldTimestamp.getTime() +
//                                    " Content: " + fieldText);
//                        }
//                    }
                        Set<KGDouble> currDoubles = new HashSet<KGDouble>();
                        snapshotParser.parseSnapshotExtractor(currDoubles, jsonSnapshot);
//                        System.out.println("Length of currDoubles: " + currDoubles.size());
                        triplesToTempQualifiers.putIfAbsent(fieldTitleQid, new HashMap<>());
                        subjectToRevisions.putIfAbsent(fieldTitleQid, new ArrayList<>());
                        subjectToRevisions.get(fieldTitleQid).add(new KGRevisionDouble(currDoubles, fieldTimestamp.getTime()));
                    }
                }
            }
        }

        // writes the rest, what is remaining, also flushes
        csvData = new ArrayList<String[]>();

        for (Map.Entry<String, HashMap<KGDouble, ArrayList<long[]>>>
                kgEntityEntry : triplesToHistory.entrySet()) {
            HashMap<KGDouble, ArrayList<long[]>> currDoubleEntries =
                    kgEntityEntry.getValue();
            String tripleSubject = kgEntityEntry.getKey();
            for (Map.Entry<KGDouble, ArrayList<long[]>>
                    kgDoubleEntry : currDoubleEntries.entrySet()) {
                ArrayList<long[]> kgDoubleActions = kgDoubleEntry.getValue();
                KGDouble kgDoubleInt = kgDoubleEntry.getKey();
                String tripleRelation = kgDoubleInt.getPropertyId();
                String tripleObject = kgDoubleInt.getObjectQid();
                String type = kgDoubleInt.getType();
                StringBuilder stringTimestamps = new StringBuilder();
                for (long[] currAction : kgDoubleActions) {

                    stringTimestamps.append(currAction[0])
                            .append(":")
                            .append((currAction[1] == 0) ? "D" : "A")
                            .append(",");
                }
                TempQualifierV2 tempQualifier =
                        triplesToTempQualifiers.get(tripleSubject).get(kgDoubleInt);

                for (String currTimestamp : tempQualifier.getTemporalQualifiers()) {
                    stringTimestamps.append(currTimestamp)
                            .append(",");
                }

                csvData.add(new String[]{
                        tripleSubject,
                        tripleRelation,
                        tripleObject,
                        type,
                        stringTimestamps.toString()
                });
            }
        }
        System.out.println("WRITING csvData FINAL");
        csvWriterTriples.writeAll(csvData);
        csvWriterEntities.writeAll(csvEntityData);

//        readDoubles = new HashSet<KGDoubleInt>();
        csvWriterTriples.flush();
        csvWriterEntities.flush();
        csvWriterTriples.close();
        csvWriterEntities.close();
    }

    @Override
    public void run() {
        // TODO: take into account :
        //  1. to filter the namespace to be in 0 (ns attribute)
        //  2.
        //  page (list):
        //    --> title (Q30098000)
        //    --> ns (0)
        //    --> id (31710113)
        //    --> revision (list):
        //      --> id (492853164)
        //      --> contributor (username, id, ip?)
        //      --> comment
        //      --> model
        //      --> format
        //      --> text
        Path pathTriples = Paths.get(outputPathTriples);
        try {
            Files.createDirectories(pathTriples);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        Path pathEntities = Paths.get(outputPathEntities);
        try {
            Files.createDirectories(pathEntities);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        File currFile = new File(this.pathToInputFile);
        Path currPath = currFile.toPath();
        final int len = currPath.getNameCount();
        String fileName = String.valueOf(currPath.subpath(len - 1, len));
        fileName = fileName.substring(0, fileName.length() - 3);
        File outputFile = new File(String.valueOf(pathTriples), fileName + ".csv");
        File outputFileEntities = new File(String.valueOf(pathEntities), fileName + ".csv");
        System.out.println("outputFile is: " + outputFile + " obtained from " +
                this.pathToInputFile);
        try {
            XMLInputFactory xmlInputFactory = XMLInputFactory.newInstance();

            if (this.fileFormat == FileFormat.SEVEN_ZIP) {
                SevenZFile archiveFile = new SevenZFile(currFile);
                SevenZArchiveEntry entry;
                while ((entry = archiveFile.getNextEntry()) != null) {
                    if (!entry.isDirectory()) {
                        try (InputStream inputStream = archiveFile.getInputStream(entry)) {
                            processStream(inputStream, outputFile, outputFileEntities, xmlInputFactory);
                        }
                    }
                }
            } else if (this.fileFormat == FileFormat.BZ2) {
                try (InputStream fis = new FileInputStream(this.pathToInputFile);
                     BZip2CompressorInputStream bzIn =
                             new BZip2CompressorInputStream(fis, true)) {  // ← true = decompress concatenated streams
                    processStream(bzIn, outputFile, outputFileEntities, xmlInputFactory);
                }
            }

        } catch (XMLStreamException | IOException | ParseException e) {
            throw new RuntimeException(e);
        }

//        try {
//
//            XMLInputFactory xmlInputFactory = XMLInputFactory.newInstance();
//
//            if (this.fileFormat == FileFormat.SEVEN_ZIP) {
//                SevenZFile archiveFile = new SevenZFile(currFile);
//                SevenZArchiveEntry entry;
//                while ((entry = archiveFile.getNextEntry()) != null) {
//                    String name = entry.getName();
//                    if (entry.isDirectory()) {
//                        System.out.printf("Found directory entry %s%n", name);
//                    } else {
//                        InputStream inputStream = archiveFile.getInputStream(entry);
//                        processStream(inputStream, outputFile, outputFileEntities, xmlInputFactory);
//                    }
//                }
//            } else {
//                BZip2CompressorInputStream inputStream =
//                        new BZip2CompressorInputStream(new FileInputStream(this.pathToInputFile));
//                processStream(inputStream, outputFile, outputFileEntities, xmlInputFactory);
//            }
//        } catch (XMLStreamException | IOException | ParseException e) {
//            throw new RuntimeException(e);
//        }
        System.out.println("=========== done with the followingfile: " + pathToInputFile);
    }
}