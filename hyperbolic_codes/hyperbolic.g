## Run Manually:
## ChangeDirectoryCurrent("C:/Users/mark/Dropbox/PhD/dev/CSSLO/hyperbolic_codes");
## Read("./hyperbolic.g");
## 
## Run via shell script
## nohup bash hyperbolic.sh&

Mat2Str := function(A)
    local a, temp;
    temp := [];
    for a in A do
        Add(temp,JoinStringsWithSeparator(List(a,String),""));
    od;
    return JoinStringsWithSeparator(temp,"\n");
end;

CosetAdj := function(C1,C2)
    local n1, n2, A, i, j;
    n1 := Length(C1);
    n2 := Length(C2);
    A := NullMat(n1,n2);
    for i in [1..n1] do
        for j in [1..n2] do
            if not IsEmpty(Intersection(C1[i],C2[j])) then
                A[i][j] := 1;
            fi;
        od;
    od;
    return A;
end;

getAdj := function(Q)
    local a, b, c, Gab, Gac, Gbc, Faces, Edges, Vertices, temp;
    a:= Q.1;
    b:= Q.2;
    c:= Q.3;
    temp := [];
    if (a <> () and b <> () and c <> () and a*b <> () and b * c <> ()) then
        Gab := SubgroupNC( Q, [a,b] );
        Gac := SubgroupNC( Q, [a,c] );
        Gbc := SubgroupNC( Q, [b,c] );
        Faces := RightCosetsNC(Q,Gab);
        Edges := RightCosetsNC(Q,Gac);
        Vertices := RightCosetsNC(Q,Gbc);
        Add(temp,Mat2Str(CosetAdj(Vertices,Edges)));
        Add(temp,Mat2Str(CosetAdj(Faces,Edges)));
        Add(temp, Mat2Str(CosetAdj(Faces,Vertices)));
    fi;
    return temp;
end;

## non-orientable
FG := FreeGroup( "a", "b", "c" );
r := 13; 
s := 13;
## Relations in terms of a,b,c, rho, sig
a:=FG.1;
b:=FG.2;
c:=FG.3;
rho:= a*b;
sig:= b*c;
rot:=[a^2, b^2, c^2, rho ^ r, sig ^ s, (a * c) ^ 2];
Grs := FG/rot;

################################################################################
## search for Gamma up to index MaxIx
################################################################################
MaxIx := 4000;
LoadPackage("LINS");
LoadPackage("json");
fileout := Concatenation([String(r),"-",String(s),"-codes.txt"]);
fn:=Filename( DirectoryCurrent( ), fileout );
# AppendTo(fn,StringFormatted("\n\n## [{},{}] Hyperbolic Tesselation",r,s));
for SG in List(LowIndexNormalSubgroupsSearchForAll(Grs, MaxIx)) do
    SG := Grp(SG);
    gens := GeneratorsOfGroup(SG);
    genStr := String(gens);
    DataList := getAdj(Grs/SG);
    if Length(DataList) > 0 then
        ix := Index(Grs,SG);
        jsonData := rec( r := r, s := s, index := ix, genStr := genStr, zEV := DataList[1], zEF := DataList[2], zFV := DataList[3] );
        AppendTo(fn,"\n", GapToJsonString(jsonData));
    fi;
od;