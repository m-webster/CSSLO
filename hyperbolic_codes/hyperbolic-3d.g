## Run Manually:
## ChangeDirectoryCurrent("C:/Users/mark/Dropbox/PhD/dev/CSSLO/hyperbolic_codes");
## Read("./hyperbolic-3d.g");
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
    local a, b, c, d, Gab, Gbc, Gcd, Gda, Faces, Edges, Vertices, temp;
    a:= Q.1;
    b:= Q.2;
    c:= Q.3;
    d:= Q.4;
    temp := [];
    if (a <> () and b <> () and c <> () and d <> () and a * b <> () and b * c <> () and c * d <> ()) then
        Gab := SubgroupNC( Q, [a,b] );
        Gab := RightCosetsNC(Q,Gab);
        Gbc := SubgroupNC( Q, [b,c] );
        Gbc := RightCosetsNC(Q,Gbc);
        Gcd := SubgroupNC( Q, [c,d] );
        Gcd := RightCosetsNC(Q,Gcd);
        Gda := SubgroupNC( Q, [d , a] );
        Gda := RightCosetsNC(Q,Gda);
        Add(temp,Mat2Str(CosetAdj(Gab,Gbc)));
        Add(temp,Mat2Str(CosetAdj(Gbc,Gcd)));
        Add(temp,Mat2Str(CosetAdj(Gcd,Gda)));
        # Add(temp,Mat2Str(CosetAdj(Vertices,Edges)));
        # Add(temp,Mat2Str(CosetAdj(Faces,Edges)));
        # Add(temp, Mat2Str(CosetAdj(Faces,Vertices)));
    fi;
    return temp;
end;

VFAdj := function(genix, SG, Grs)
    local FGens, FSG, VList, n, FList, m, A, i, j, FRep, VF;
    ## Vertices
    VList := RightTransversal(Grs,SG);
    n := Length(VList);
    ## Faces 
    FGens := Concatenation (GeneratorsOfGroup(Grs){genix},GeneratorsOfGroup(SG));
    FSG := Subgroup(Grs, FGens);
    FList := RightCosets(Grs, FSG);
    m := Length(FList);
    ## Adjacency Matrix
    A := NullMat(m,n);
    for i in [1..m] do
        FRep := Representative(FList[i]);
        for VF in RightCosets(FSG, SG) do
            j := PositionCanonical(VList, FRep * Representative(VF));
            A[i,j] := 1;
        od;
    od;
    return A;
end;

getCellsOld := function(SG, Grs,k)
    ## cells made from combinations of k generators
    local s, temp;
    temp := [];
    for s in Combinations([1,2,3,4],k) do
        Append(temp,VFAdj(s, SG, Grs));
    od;
    return temp;
end;

getCells := function(SG, Grs,k)
    ## cells made from combinations of k generators
    local m, n, s, A, temp, GList, Q, FGens, FList, i, j;
    Q := Grs/SG;
    GList := List(Q);
    n := Length(GList);
    temp := [];
    for s in Combinations([1,2,3,4],k) do
        FGens := GeneratorsOfGroup(Q){s};
        FList := RightCosets(Q, SubgroupNC(Q,FGens));
        m := Length(FList);
        A := NullMat(m,n);
        for i in [1..m] do
            for j in [1..n] do
                if GList[j] in FList[i] then
                    A[i,j] := 1;
                fi;
            od;
        od;
        Append(temp,A);
    od;
    return temp;
end;

## non-orientable
FG := FreeGroup( "a", "b", "c", "d" );
rab := 3; 
rbc := 3;
rcd := 3;
rda := 3;

rac := 2;
rbd := 2;


## Relations in terms of a,b,c, rho, sig
a:=FG.1;
b:=FG.2;
c:=FG.3;
d:=FG.4;

rot:=[a^2, b^2, c^2, d^2, (a * b) ^ rab, (b * c)  ^ rbc, (c * d)  ^ rcd, (d * a) ^ rda, (a * c) ^ rac, (b * d) ^ rbd];
Grs := FG/rot;

################################################################################
## search for Gamma up to index MaxIx
################################################################################
MinIx := 1;
MaxIx := 2000;
LoadPackage("LINS");
LoadPackage("json");
fileout := Concatenation([String(rab),"-",String(rbc),"-",String(rcd),"-",String(rda),"-codes.txt"]);
fn:=Filename( DirectoryCurrent( ), fileout );
# AppendTo(fn,StringFormatted("\n\n## [{},{}] Hyperbolic Tesselation",r,s));
for SG in List(LowIndexNormalSubgroupsSearchForAll(Grs, MaxIx)) do
    SG := Grp(SG);
    ix := Index(Grs,SG);
    if ix > MinIx then
        gens := GeneratorsOfGroup(SG);
        genStr := String(gens);
        # AppendTo(fn, "\n", Index(Grs,SG), genStr);
        DataList := getAdj(Grs/SG);
        SX := getCells(SG, Grs,3);
        SZ := getCells(SG, Grs,2);
        if Length(SX) > 0 then
            n := Length(SX[1]);
            
            jsonData := rec( N := n, index := ix, genStr := genStr, zSX := Mat2Str(SX), zSZ := Mat2Str(SZ) );
            AppendTo(fn,"\n", GapToJsonString(jsonData));
        fi;
    fi;
od;