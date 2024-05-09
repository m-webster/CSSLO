## Run Manually:
## ChangeDirectoryCurrent("/Users/markwebster/Dropbox/Quantum Computing/dev/CSSLO/hyperbolic_codes");
## Read("./RGComplex.g");
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

CosetAdj := function(C1,C2,Incl)
    local n1, n2, A, i, j;
    n1 := Length(C1);
    n2 := Length(C2);
    A := NullMat(n1,n2);
    for i in [1..n1] do
        for j in [1..n2] do
            if (not Incl and not IsEmpty(Intersection(C1[i],C2[j]))) or (Incl and IsSubsetSet(List(C2[j]),List(C1[i])) ) then
                A[i][j] := 1;
            fi;
        od;
    od;
    return A;
end;

FlagCosets := function(Q,k)
    ## cell cosets made from combinations of k generators
    local s, temp, QGens, kGens;
    QGens := GeneratorsOfGroup(Q);
    temp := [];
    for s in Combinations([1..Length(QGens)],k) do
        kGens := QGens{s};
        Append(temp,RightCosets(Q, SubgroupNC(Q,kGens)));
    od;
    return temp;
end;

FlagBoundaryOps := function(Q,D)
    local Curr, Prev, temp, k;
    Prev := FlagCosets(Q,D);
    temp := [];
    for k in [1..D] do
        Curr := FlagCosets(Q,D-k);
        Add(temp, CosetAdj(Curr,Prev,true));
        Prev := Curr;
    od;
    return temp;
end;

## Breuckmann Method
RGCosets := function(Q,k)
    ## cell cosets made from all generators, apart from the kth one
    local s, QGens, kGens;
    QGens := GeneratorsOfGroup(Q);
    s := [1..Length(QGens)];
    Remove(s,k);
    kGens := QGens{s};
    return RightCosets(Q, SubgroupNC(Q,kGens));
end;

RGBoundaryOps := function(Q,D)
    local Curr, Prev, temp, k;
    Prev := RGCosets(Q,1);
    temp := [];
    for k in [1..D] do
        Curr := RGCosets(Q,k+1);
        Add(temp, CosetAdj(Curr,Prev,false));
        Prev := Curr;
    od;
    return temp;
end;

## Default Coxeter matrix - lower triagonal, diagonal all ones, off diagonals are 2
coxeterMat := function(D)
    local R, i, j, Ri;
    R := IdentityMat(D+1);
    for i in [1..D+1] do
        for j in [1..i-1] do
            R[i,j] := 2;
        od;
    od;
    return R;
end;

## Generate Coxeter group from Coxeter matrix
coxeterGroup := function(R,D)
    local FG, FGGens, Rels, i, j;
    FG := FreeGroup(D+1);
    FGGens := GeneratorsOfGroup(FG);
    Rels := [];
    for i in [1..D+1] do
        for j in [1..i] do
            Add(Rels,(FGGens[i] * FGGens[j])^R[i][j]);
        od;
    od;
    return FG/Rels;
end;

myType := "FG"; ## Flag Complex 
myType := "RG"; ## Reflection Group Complex

#myRels := [3,3,3,3]; ## for 3D complex 
myRels := [3,6];     ## for 2D complex 

## min and max group indices for normal subgroup search
MinIx := 6;
MaxIx := 4000;

# Automatically create Coxeter Matrix based on myRels list - can also be set up manually if desired
if Length(myRels) = 2 then
    D:= 2;
    R := coxeterMat(D);
    R[2,1] := myRels[1];
    R[3,2] := myRels[2];
    fileout := StringFormatted("{}-{}-{}.txt",myType,myRels[1],myRels[2]);
else
    D := 3;
    R := coxeterMat(D);
    R[2,1] := myRels[1];
    R[3,2] := myRels[2];
    R[4,3] := myRels[3];
    R[4,1] := myRels[4];
    fileout := StringFormatted("{}-{}-{}-{}-{}.txt",myType,myRels[1],myRels[2],myRels[3],myRels[4]);
fi;

PrintArray(R);
Grs := coxeterGroup(R,D);

################################################################################
## search for Gamma up to index MaxIx
################################################################################

LoadPackage("LINS");
LoadPackage("json");

fn:=Filename( DirectoryCurrent( ), fileout );
AppendTo(fn,"################################################################\n");
AppendTo(fn,StringFormatted("{}-Dimensional {} Complexes\n",D,myType));
AppendTo(fn,StringFormatted("Min Index {}; Max Index {}\n",MinIx,MaxIx));
AppendTo(fn,"Coxeter Matrix\n");
AppendTo(fn,StringFormatted("{}\n", Mat2Str(R)));
AppendTo(fn,"################################################################\n");
for SG in List(LowIndexNormalSubgroupsSearchForAll(Grs, MaxIx)) do
    SG := Grp(SG);
    ix := Index(Grs,SG);
    Q := Grs/SG;
    if ix > MinIx then
        Print(ix);
        Print("\n");
        jsonData := rec( index := ix );
        if myType = "FG" then
            temp := FlagBoundaryOps(Q,D);
        else 
            temp := RGBoundaryOps(Q,D);
        fi;
        for k in [1..Length(temp)] do
            jsonData.(StringFormatted("Z{}", k)) := Mat2Str(temp[k]);
        od;
        AppendTo(fn,"\n", GapToJsonString(jsonData));
    fi;
od;
