function value = sha256File(path)
%SHA256FILE Compute a binary file SHA-256 using MATLAB's Java runtime.
arguments
    path (1,1) string
end
if ~isfile(path), error("RetinaSathi:MissingArtifact", "File not found: %s", path); end
digest = java.security.MessageDigest.getInstance("SHA-256");
file = fopen(path, "r");
if file < 0, error("RetinaSathi:MissingArtifact", "Cannot open: %s", path); end
cleanup = onCleanup(@() fclose(file)); %#ok<NASGU>
while true
    chunk = fread(file, 1024 * 1024, "*uint8");
    if isempty(chunk), break; end
    digest.update(typecast(chunk, "int8"));
end
bytes = typecast(int8(digest.digest()), "uint8");
value = lower(string(reshape(dec2hex(bytes, 2).', 1, [])));
end
