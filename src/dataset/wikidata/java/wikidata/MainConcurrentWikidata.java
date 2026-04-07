package wikidata;
// This module is very fast parallel reader of all the history of the wikidata and aggregating this history in a
// single graph. I think it will be key to get continual representation of the graph.

import wikidata.misc.JSONSnapshot;
import wikidata.misc.ReaderJSONStream;

import java.io.IOException;
import java.nio.file.FileVisitOption;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import java.util.stream.Stream;

class Display extends Thread {
    private final BlockingQueue<JSONSnapshot> linkedBlockingQueue;
    //     private static AtomicInteger counter;
    private AtomicLong nrProcessedEntities;
    private long startingTime;

    private long lastCheckedTimeNumber;

    private boolean stopRequested = false;

    private ExecutorService executorServiceReader = null;

    private ExecutorService executorServiceParser = null;

    private ThreadPoolExecutor threadPoolReader = null;
    private ThreadPoolExecutor threadPoolParser = null;

    private long howMuchSleep = 60000;

    public Display(BlockingQueue<JSONSnapshot> linkedBlockingQueue, AtomicLong nrProcessedEntities,
                   ExecutorService executorServiceReader, ExecutorService executorServiceParser,
                   ThreadPoolExecutor threadPoolReader, ThreadPoolExecutor threadPoolParser) {
        this.nrProcessedEntities = nrProcessedEntities;
        this.linkedBlockingQueue = linkedBlockingQueue;
        this.startingTime = System.currentTimeMillis();
        this.executorServiceParser = executorServiceParser;
        this.executorServiceReader = executorServiceReader;
        this.threadPoolReader = threadPoolReader;
        this.threadPoolParser = threadPoolParser;
    }

    public synchronized void requestStop() {
        this.stopRequested = true;
    }

    public synchronized boolean isStopRequested() {
        return this.stopRequested;
    }

    public void run() {
        while (!this.stopRequested) {
            try {
                long currNr = this.nrProcessedEntities.get();
                long currTimeInMillis = System.currentTimeMillis();
                long diffTimes = currTimeInMillis - this.startingTime;
                double diffTimeMins = (double) (diffTimes / 1000.0 / 60.0);
                double avgPerMin = currNr / diffTimeMins;

                double currAvgPerMin = (currNr - lastCheckedTimeNumber) / (double) (howMuchSleep / 1000.0 / 60.0);
                System.out.println(avgPerMin + " entities per min; " +
                        currAvgPerMin + " current entities per min; " +
                        "nr of processed entities: " + currNr +
                        " threadPoolReader active: " + threadPoolReader.getActiveCount());
                lastCheckedTimeNumber = currNr;

                if (threadPoolReader.getActiveCount() == 0) {
                    threadPoolReader.shutdown();
                }
                Thread.sleep(howMuchSleep);
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
    }
}

public class MainConcurrentWikidata {
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

        String pathTo7zHistoryFiles = "";
        if (args.length > 0) {
            pathTo7zHistoryFiles = args[0];
        } else {
            pathTo7zHistoryFiles = "/path/to/data";
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

        System.out.println("pathTo7zHistoryFiles set to: " + pathTo7zHistoryFiles);
        System.out.println("nr of threads file reader set to: " + nrThreadsReadFiles);

        ThreadPoolExecutor threadPoolReader = new ThreadPoolExecutor(nrThreadsReadFiles, 100000,
                395, TimeUnit.DAYS, new LinkedBlockingQueue<>(100000));

        List<Path> allFilesInDir = getAllFilesInDir(pathTo7zHistoryFiles);
        Collections.shuffle(allFilesInDir, new Random(5437));

        AtomicLong nrProcessedEntities = new AtomicLong(0);
        for (Path currDir : allFilesInDir) {
            String currDirStr = currDir.toString();
            if (!currDirStr.endsWith(".7z")) {
                System.out.println("ignoring: " + currDirStr);
                continue;
            }

            threadPoolReader.execute(new ReaderJSONStream(currDirStr,
                    null,
                    nrProcessedEntities,
                    outputPathTriples,
                    outputPathEntities,
                    ReaderJSONStream.FileFormat.SEVEN_ZIP));
        }

        Display displayStats = new Display(null, nrProcessedEntities,
                null, null,
                threadPoolReader, null);
        displayStats.setDaemon(true);
        displayStats.start();
    }
}
