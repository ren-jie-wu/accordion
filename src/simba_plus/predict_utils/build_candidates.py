def build_candidate_peaks(adata_CP, snp_hits, region_hits):
    """
    Create the final unified set of peaks:
        - peaks from SNP overlap
        - peaks from region overlap
        - if none provided → all peaks
    """
    if not snp_hits and not region_hits:
        print("No SNPs or regions provided → using all peaks.")
        return list(adata_CP.var_names)

    peaks = set(snp_hits) | set(region_hits)
    return list(peaks)
