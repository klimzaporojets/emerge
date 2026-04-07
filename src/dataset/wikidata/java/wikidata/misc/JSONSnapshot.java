package wikidata.misc;

import java.util.Date;

public class JSONSnapshot {
    public JSONSnapshot(String qid, String text, String comment) {
        this(qid, text, comment, null);
    }

    public JSONSnapshot(String qid, String text, String comment, Date timestamp) {
        this.qid = qid;
        this.text = text;
        this.comment = comment;
        this.timestamp = timestamp;
    }

    private String qid;
    private String text;
    //    private String date;
    private String comment;

//    private String timestamp;
    private Date timestamp;

    public Date getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(Date timestamp) {
        this.timestamp = timestamp;
    }

    public String getQid() {
        return qid;
    }

    public void setQid(String qid) {
        this.qid = qid;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }

    @Override
    public String toString() {
        return "JSONSnapshot{" +
                "qid='" + qid + '\'' +
                ", text='" + text + '\'' +
                ", comment='" + comment + '\'' +
                ", timestamp='" + timestamp + '\'' +
                '}';
    }

    public String getComment() {
        return comment;
    }

    public void setComment(String comment) {
        this.comment = comment;
    }
}