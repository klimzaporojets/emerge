package wikidata.misc;

import java.util.Objects;
import java.util.Set;

// object that associate to a set of doubles a timestamp
public class KGRevisionDouble {
    private Set<KGDouble> kgDouble;
    private long timestamp;

    public KGRevisionDouble(Set<KGDouble> kgDouble, long timestamp) {
        this.kgDouble = kgDouble;
        this.timestamp = timestamp;
    }


//    private String type;
//
//    private TempQualifierV2 tempQualifier;

//    private long timestamp; // timestamp of when this kg triple was created.
//    private boolean isQualifier;
//
//    public long getTimestamp() {

    public Set<KGDouble> getKgDouble() {
        return kgDouble;
    }

    public void setKgDouble(Set<KGDouble> kgDouble) {
        this.kgDouble = kgDouble;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }
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

//    public TempQualifierV2 getTempQualifier() {
//        return tempQualifier;
//    }
//
//    public void setTempQualifier(TempQualifierV2 tempQualifier) {
//        this.tempQualifier = tempQualifier;
//    }
//
//    public String getType() {
//        return type;
//    }
//
//    public void setType(String type) {
//        this.type = type;
//    }
//
//    public String getPropertyId() {
//        return propertyId;
//    }
//
//    public void setPropertyId(String propertyId) {
//        this.propertyId = propertyId;
//    }
//
//    public String getObjectQid() {
//        return objectQid;
//    }
//
//    public void setObjectQid(String objectQid) {
//        this.objectQid = objectQid;
//    }
//
//    public KGRevisionDouble(String propertyId, String objectQid, String type) {
//        this.propertyId = propertyId;
//        this.objectQid = objectQid;
//        this.type = type;
//    }
//
//    @Override
//    public boolean equals(Object o) {
//        if (this == o) return true;
//        if (o == null || getClass() != o.getClass()) return false;
//        KGRevisionDouble that = (KGRevisionDouble) o;
//        return Objects.equals(propertyId, that.propertyId)
//                && Objects.equals(objectQid, that.objectQid)
//                && Objects.equals(type, that.type);
//    }
//
//    @Override
//    public int hashCode() {
//        return Objects.hash(propertyId, objectQid, type);
//    }
//
//    @Override
//    public String toString() {
//        return "KGTripleInt{" +
//                ", propertyId=" + propertyId +
//                ", objectQid=" + objectQid +
//                '}';
//    }
}