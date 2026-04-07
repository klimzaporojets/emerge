package wikidata.misc;

import javax.xml.crypto.Data;
import java.util.Date;
import java.util.Objects;

public class KGTriple {
    private String subjectQid;
    private String propertyId;
    private String objectQid;
    private long timestamp; // timestamp of when this kg triple was created.
    private boolean isQualifier;

    public long getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }

    public boolean isQualifier() {
        return isQualifier;
    }

    public void setQualifier(boolean qualifier) {
        isQualifier = qualifier;
    }


    public String getSubjectQid() {
        return subjectQid;
    }

    public void setSubjectQid(String subjectQid) {
        this.subjectQid = subjectQid;
    }

    public String getPropertyId() {
        return propertyId;
    }

    public void setPropertyId(String propertyId) {
        this.propertyId = propertyId;
    }

    public String getObjectQid() {
        return objectQid;
    }

    public void setObjectQid(String objectQid) {
        this.objectQid = objectQid;
    }

    public KGTriple(String subjectQid, String propertyId, String objectQid,
                    boolean isQualifier, Date timestamp) {
        this.subjectQid = subjectQid;
        this.propertyId = propertyId;
        this.objectQid = objectQid;
        this.isQualifier = isQualifier;
        this.timestamp = timestamp.getTime();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        KGTriple kgTriple = (KGTriple) o;
        return this.subjectQid.equals(kgTriple.subjectQid) &&
                this.propertyId.equals(kgTriple.propertyId) &&
                this.objectQid.equals(kgTriple.objectQid) &&
                this.isQualifier == kgTriple.isQualifier;
    }

    @Override
    public int hashCode() {
        return Objects.hash(subjectQid, propertyId, objectQid, isQualifier);
    }
}