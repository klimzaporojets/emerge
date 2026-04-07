package wikidata;
// This module is very fast parallel reader of all the history of the wikidata and aggregating this history in a
// single graph. I think it will be key to get continual representation of the graph.

import wikidata.misc.ReaderJSONStream;

import java.io.IOException;
import java.nio.file.FileVisitOption;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class MainConcurrentWikidataBZ2 {
    static final int MAX_QUEUE_SIZE = 1000;

    public static List<Path> getAllFilesInDir(String root_dir) {
        try {
            Stream<Path> stream = Files.walk(Paths.get(root_dir), FileVisitOption.FOLLOW_LINKS);
            stream = stream.filter(Files::isRegularFile);
            return stream.collect(Collectors.toList());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void main(String[] args) throws InterruptedException, IOException {
        System.setProperty("jdk.xml.totalEntitySizeLimit", String.valueOf(Integer.MAX_VALUE));

        String pathToBZ2Files = "";
        if (args.length > 0) {
            pathToBZ2Files = args[0];
        } else {
            pathToBZ2Files = "/path/to/data";
        }

        int nrThreadsReadFiles = 10;

        String outputPathTriples = "/path/to/data";
        if (args.length > 1) {
            outputPathTriples = args[1];
        }

        String outputPathEntities = "/path/to/data";
        if (args.length > 2) {
            outputPathEntities = args[2];
        }

        if (args.length > 3) {
            nrThreadsReadFiles = Integer.parseInt(args[3]);
        }

        System.out.println("pathToBZ2Files set to: " + pathToBZ2Files);
        System.out.println("nr of threads file reader set to: " + nrThreadsReadFiles);

        ThreadPoolExecutor threadPoolReader = new ThreadPoolExecutor(nrThreadsReadFiles, 100000,
                395, TimeUnit.DAYS, new LinkedBlockingQueue<>(100000));

        List<Path> allFilesInDir = getAllFilesInDir(pathToBZ2Files);
        Collections.shuffle(allFilesInDir, new Random(5437));

        AtomicLong nrProcessedEntities = new AtomicLong(0);
        for (Path currDir : allFilesInDir) {
            String currDirStr = currDir.toString();
            if (!currDirStr.endsWith(".bz2")) {
                System.out.println("ignoring: " + currDirStr);
                continue;
            }

            threadPoolReader.execute(new ReaderJSONStream(currDirStr,
                    null,
                    nrProcessedEntities,
                    outputPathTriples,
                    outputPathEntities,
                    ReaderJSONStream.FileFormat.BZ2));
        }

        Display displayStats = new Display(null, nrProcessedEntities,
                null, null,
                threadPoolReader, null);
        displayStats.setDaemon(true);
        displayStats.start();
    }
}
