package wikidata.misc;

import java.util.Objects;

public class KGTripleInt {
    private int predicateQid;
    private int propertyId;
    private int objectQid;

    public int getPredicateQid() {
        return predicateQid;
    }

    public void setPredicateQid(int predicateQid) {
        this.predicateQid = predicateQid;
    }

    public int getPropertyId() {
        return propertyId;
    }

    public void setPropertyId(int propertyId) {
        this.propertyId = propertyId;
    }

    public int getObjectQid() {
        return objectQid;
    }

    public void setObjectQid(int objectQid) {
        this.objectQid = objectQid;
    }

    public KGTripleInt(int predicateQid, int propertyId, int objectQid) {
        this.predicateQid = predicateQid;
        this.propertyId = propertyId;
        this.objectQid = objectQid;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        KGTripleInt kgTriple = (KGTripleInt) o;
        return this.predicateQid == kgTriple.predicateQid &&
                this.propertyId == kgTriple.propertyId &&
                this.objectQid == kgTriple.objectQid;
    }

    @Override
    public int hashCode() {
        return Objects.hash(predicateQid, propertyId, objectQid);
    }

    @Override
    public String toString() {
        return "KGTripleInt{" +
                "predicateQid=" + predicateQid +
                ", propertyId=" + propertyId +
                ", objectQid=" + objectQid +
                '}';
    }
}