package wikidata.misc;

import java.util.ArrayList;
import java.util.List;

public class TempQualifierV2 {
    // The difference with TempQualifierV2 is that here only 1 list is used for all
    // the temporal qualifiers. The code of the qualifier is inherited from Wikidata
    // and is not separated into different categories.

    List<String> temporalQualifiers = new ArrayList<String>();

    public List<String> getTemporalQualifiers() {
        return temporalQualifiers;
    }

    public void addTemporalQualifier(String temporalQualifier) {
        this.temporalQualifiers.add(temporalQualifier);
    }

}
