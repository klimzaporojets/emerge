package wikidata.misc;

import java.util.Objects;

public class KGDouble {
    private String propertyId;
    private String objectQid;

    private String type;

    private TempQualifierV2 tempQualifier;

//    private long timestamp; // timestamp of when this kg triple was created.
//    private boolean isQualifier;
//
//    public long getTimestamp() {
//        return timestamp;
//    }
//
//    public void setTimestamp(long timestamp) {
//        this.timestamp = timestamp;
//    }
//
//    public boolean isQualifier() {
//        return isQualifier;
//    }
//
//    public void setQualifier(boolean qualifier) {
//        isQualifier = qualifier;
//    }

    public TempQualifierV2 getTempQualifier() {
        return tempQualifier;
    }

    public void setTempQualifier(TempQualifierV2 tempQualifier) {
        this.tempQualifier = tempQualifier;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
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

    public KGDouble(String propertyId, String objectQid, String type) {
        this.propertyId = propertyId;
        this.objectQid = objectQid;
        this.type = type;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        KGDouble that = (KGDouble) o;
        return Objects.equals(propertyId, that.propertyId)
                && Objects.equals(objectQid, that.objectQid)
                && Objects.equals(type, that.type);
    }

    @Override
    public int hashCode() {
        return Objects.hash(propertyId, objectQid, type);
    }

    @Override
    public String toString() {
        return "KGTripleInt{" +
                ", propertyId=" + propertyId +
                ", objectQid=" + objectQid +
                '}';
    }
}