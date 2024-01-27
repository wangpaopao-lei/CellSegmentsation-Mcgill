#!/usr/bin/env python
#
# Hi There!
#
# You may be wondering what this giant blob of binary data here is, you might
# even be worried that we're up to something nefarious (good for you for being
# paranoid!). This is a base85 encoding of a zip file, this zip file contains
# an entire copy of pip (version 23.2.1).
#
# Pip is a thing that installs packages, pip itself is a package that someone
# might want to install, especially if they're looking to run this get-pip.py
# script. Pip has a lot of code to deal with the security of installing
# packages, various edge cases on various platforms, and other such sort of
# "tribal knowledge" that has been encoded in its code base. Because of this
# we basically include an entire copy of pip inside this blob. We do this
# because the alternatives are attempt to implement a "minipip" that probably
# doesn't do things correctly and has weird edge cases, or compress pip itself
# down into a single file.
#
# If you're wondering how this is created, it is generated using
# `scripts/generate.py` in https://github.com/pypa/get-pip.

import sys

this_python = sys.version_info[:2]
min_version = (3, 7)
if this_python < min_version:
    message_parts = [
        "This script does not work on Python {}.{}".format(*this_python),
        "The minimum supported Python version is {}.{}.".format(*min_version),
        "Please use https://bootstrap.pypa.io/pip/{}.{}/get-pip.py instead.".format(*this_python),
    ]
    print("ERROR: " + " ".join(message_parts))
    sys.exit(1)


import os.path
import pkgutil
import shutil
import tempfile
import argparse
import importlib
from base64 import b85decode


def include_setuptools(args):
    """
    Install setuptools only if absent and not excluded.
    """
    cli = not args.no_setuptools
    env = not os.environ.get("PIP_NO_SETUPTOOLS")
    absent = not importlib.util.find_spec("setuptools")
    return cli and env and absent


def include_wheel(args):
    """
    Install wheel only if absent and not excluded.
    """
    cli = not args.no_wheel
    env = not os.environ.get("PIP_NO_WHEEL")
    absent = not importlib.util.find_spec("wheel")
    return cli and env and absent


def determine_pip_install_arguments():
    pre_parser = argparse.ArgumentParser()
    pre_parser.add_argument("--no-setuptools", action="store_true")
    pre_parser.add_argument("--no-wheel", action="store_true")
    pre, args = pre_parser.parse_known_args()

    args.append("pip")

    if include_setuptools(pre):
        args.append("setuptools")

    if include_wheel(pre):
        args.append("wheel")

    return ["install", "--upgrade", "--force-reinstall"] + args


def monkeypatch_for_cert(tmpdir):
    """Patches `pip install` to provide default certificate with the lowest priority.

    This ensures that the bundled certificates are used unless the user specifies a
    custom cert via any of pip's option passing mechanisms (config, env-var, CLI).

    A monkeypatch is the easiest way to achieve this, without messing too much with
    the rest of pip's internals.
    """
    from pip._internal.commands.install import InstallCommand

    # We want to be using the internal certificates.
    cert_path = os.path.join(tmpdir, "cacert.pem")
    with open(cert_path, "wb") as cert:
        cert.write(pkgutil.get_data("pip._vendor.certifi", "cacert.pem"))

    install_parse_args = InstallCommand.parse_args

    def cert_parse_args(self, args):
        if not self.parser.get_default_values().cert:
            # There are no user provided cert -- force use of bundled cert
            self.parser.defaults["cert"] = cert_path  # calculated above
        return install_parse_args(self, args)

    InstallCommand.parse_args = cert_parse_args


def bootstrap(tmpdir):
    monkeypatch_for_cert(tmpdir)

    # Execute the included pip and use it to install the latest pip and
    # setuptools from PyPI
    from pip._internal.cli.main import main as pip_entry_point
    args = determine_pip_install_arguments()
    sys.exit(pip_entry_point(args))


def main():
    tmpdir = None
    try:
        # Create a temporary working directory
        tmpdir = tempfile.mkdtemp()

        # Unpack the zipfile into the temporary directory
        pip_zip = os.path.join(tmpdir, "pip.zip")
        with open(pip_zip, "wb") as fp:
            fp.write(b85decode(DATA.replace(b"\n", b"")))

        # Add the zipfile to sys.path so that we can import it
        sys.path.insert(0, pip_zip)

        # Run the bootstrap
        bootstrap(tmpdir=tmpdir)
    finally:
        # Clean up our temporary working directory
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


DATA = b"""
P)h>@6aWAK2ml*O_EvJ33*7hs003nH000jF003}la4%n9X>MtBUtcb8c|B0UO2j}6z0X&KUUXrd;wrc
n6ubz6s0VM$QfAw<4YV^ulDhQoop$MlK*;0e<?$L01LzdVw?IP-tnf*qTlkJj!Mom=viw7qw3H>hK(>
3ZJA0oQV`^+*aO7_tw^Cd$4zs{Pl#j>6{|X*AaQ6!2wJ?w>%d+2&1X4Rc!^r6h-hMtH_<n)`omXfA!z
c)+2_nTCfpGRv1uvmTkcug)ShEPeC#tJ!y1a)P)ln~75Jc!yqZE1Gl6K?CR$<8F6kVP)a}pU*@~6k=y
<MFxvzbFl3|p@5?5Ii7qF0_`NT{r7m1lM_B44a9>d5{IF3D`nKTt~p1QY-O00;mZO7>Q7_pjHy0RRA2
0{{RI0001RX>c!JUu|J&ZeL$6aCu!)OK;mS48HqU5b43r;JP^vOMxACEp{6QLy+m1h%E`C9MAjpBNe-
8r;{H19{ebpf{zJ27j)n8%0=-6Z#elILRo@w9oRWWbO{z8ujDS!QAC@3T%nJCf;1rX6ghzu#Z}<GSE4
4EG}J&ngovyJ$%DCh>R@K&*?Hgj1WFD91+adaM4G`4Xs@*hA^t@nbDYdL)-aOjsW~3}QVVby(8=@7U$
Fzj5Y{w!2hUUH`?e9j7WDA;>-1aos>7j{2$~BfyL8p@__Y98dsP#Bs7^<X<wp+-f{6%mc1~N!0T>lWF
=e_gr;(4^?am?Cp93+7b-!?~nb}-$cPSR1zckA*zNp!)$;YjlZrfn&RWNM}=QA7*cb8A{(9@{5!vBfq
rEMoeu5FvJZngI@N#4#(2v$WnMGCAVD?b9t8W^qDfcFBe5ZZF%dPAPaq#<aBs;+HiVj+9PK#6heH_-Q
-kVzlI0rncJH8Q{ZFBFwrpI^^9n>>ikclG~yPvCg`JUGb_W2#PdCXxx}7!|T*xc9qdnTILbO-nAJaF2
~0snMF<S>DU<%E01X4*yW9@|}F2;vY~;0|XQR000O88%p+8eg`F$&;kGeqy+!~6#xJLaA|NaUte%(a4
m9mZf<3AUtcb8d3{vDZrd;nz56RT_b?l9y`mj3AXtV0MT+&(WJ!7$x<XKFy3uA!t|VtMivIgZJ5Ji5n
+}O2laG&&&kn<Ivc;_N2)LD*FI(_y<sdV43#Nct)d~Djbf-Z=u8IOJY7eM4${JnKJ`I8;rxdD0pnokZ
%t1O(o{kB4L(#6WNXXLn@Ri9Miq52d?_ou0Rc)-Nw2hL1)Vnh{FFp1(!Y~Yi6Zr7%Cv?>|Xq_^eh*q`
qYNbl$TgcX!{RW4b=Vw*pI`moV*K|DJ2bY*KQV<MvTF2m*rdGtEu%;pm-_&W{2D2Z_Z_^twpM1Z)o=+
AqhUg-JPjL_gITiyC;k=D@`*;L!0=}(r1nNN>iviHGglIK{X_)>pN=IEr427|<0g`vfCSX-CrF6hnx-
fU6^LzLVM{GttvQ!RX(K-@qvQ<9nZh3{TwCd*xxj~wep|+d4YrpRGd3uJ(;$x#MJ^wO(dX9-I(W~SOL
|!j@ev4<Eyb3wu9PhFEUCh#7vF2;?78p&2>#PBd+t2O-3Y4TDlA%@&y9h}l?d7(gvc*a&O+atWdOv5|
XtFg8N1I1Eg2~6T^Prn{|GZSIw2~Ql9c?>!a3=lwO6eT!TZzV{RAoH`=gPAEk0OKF^-L_LxAV)%Ld>V
rC7Ea!84dqJ@cSb~%=6Dm=^V^deci#%k)qhs`k`mikNs;GRv|T<cNgr(f&zrAF^e3QMFK@rmRukK<~h
DGnzmlr2lU<HA7<r6E*!Gq-55ghPB?mHgq@`sC)hvW;?&?lxAeg2z~=5)lly_z!5`Ee^4m$72}H*Uhu
#huJ3^d6GO{;G9x1r>RB1+w&XWHK8?pSmvO+Mn5HP0Rg<yXP5hf5`O3iN%nTcXZlvCP~V}zj31%LM>&
0e2!{O&s!2A%Oz`W5|6)QOoeMptG0vVbf-p%MA<(l*rGUr<W}1QY-O00;maO7>RG$G|nf0000U0RR9D
0001RX>c!ac`kH$aAjmAj<HU~FbqZae#PBbp85|A3~X;eVm7U5EhTo8IEDN@P8ls7-*bu-NCRQBoJn^
iQAVkDRLUzpPe}~%$w)4VGpq9sQ9OsArVq@gW&td8ktF(xhi|JBx9SfJ>&U%1)EvFVxRjSzi=C>J@cM
k87yJyz4~-Qcqlg}hXv}1CF`fEox?~SG{pae%Dy$pBG>tnWs3{>FohpTZSG@fe-hAmws@4PFv7Mv`H@
JnAXTbgKqwrl)IWaYE>+%OsO9KQH0000802@m7R+hDJp-}+<06hW#02u%P0B~t=FJEbHbY*gGVQep7U
ukY>bYEXCaCvo+F;B!W42Adn3hP*|5~K?fa1xA6Cs^1JI)&D4jnX98E~x(=w{RdN$P(+xdH(X;aUMbE
La7HDOJ;>ViJroJQOYSq=f31Z#UCgsvZ;PjisC7~V50}YW@1zhN!C_4fs|i^>lX7r-W?|$V(y(g0ZOD
x-5bTWf^iasXM`rih^<v!W`vODRRPWL)$4oIy_Lw@%52^TY6ciWDVPL;9>?Sk#%z{jZl{Ri-7?Gn9_p
NH(fR_VZQx#ZustU5xCHVj%1=)fT*F;XSi#wiQR~iuoy}(RFp&L9pfC#Zn^7Ax<k&)!ljMjX4O3A89S
m#?Gl(S-mv1t5$e0@ASnWu?TZ>z>2yIKB7|@~y3-1&J5eC&FySna4hw0fjd92G^LTzc+Br>7Y7w1=({
s_3<|LwzLQl3jT^=CKlyadUgD7M{+)3>-rRJjjOO9KQH0000802@m7Rtz49V_OUW00Srh02%-Q0B~t=
FJEbHbY*gGVQepAb!lv5UuAA~E^v9(T1#`|xDmeVS714ZB@>eSIHgphB=gYhxH9p$W;~m0sZ?Bwghq@
hk_(WwwJ!hnbT<GJASv%`Dwoy462V5JA74KJ*z>%XT~X$2UELO<u8zEFStohU_O)Pztjn}5>Wbx^D5}
p)=7nt84rjpQ!t=bvqBu6SXjxf*{)}V#v6kjnleUMl*qKLJw7ma)>Zw|O-`<I|S?oo9WLaI7Jj0bG(*
*BD&IQk37g?)l+Ec^(x7Q-g_%6+Eu3@x)k0Kj_pRU%)tGDY{|G2pPA!HXV7wN9#A$tcJh3tKUi=}1AK
5}@x?izfD%tH35f>#U0v?-c6x#d+}i#X$=E%t?3;qCzPO{p3XDn-l0g8$MLf}@FhxjzhJPffk$LZTb=
tRL0mAd`8KB>SS|Ny1Wz!%10Z<UfmdQGx29X`GcsEWtz-Ff;S(hF6ImoSS3#^%FkxHfaDO;NVa_bb|K
}GCP23bBvC>P4l!(Z9X~Qr(M}5e1M{2V-3vl>e`}|vFvt@s535m*|M}OlVSM$)RrHcBrimd6?lFPUdh
^8oI-}L;caqLRJjDa?_Dr07Ysf#%z>QWYbSDW3_SKrT&dAFG`Lt`@W9KJiJ}<Pen(?|l2qvtTpBDE3$
PFaGv!_5r%NVV5wab92Q2!l!s&*fyMeK-hVh2zaA2!M6};_4#rzaz2Mp9n*;gyJW4x3cM{%XBOu>-Jm
Eim0UQMILLA#<&5?}IiA5v%!>tEItSETqsiWmt%EBta_NRXj{H*Zo{ba+L0f#Cr>zURR@B*qHa1TLRl
QIY3XdTuN;Q8cY|sQ{2jC4o$vPgD13HO~sl#?~l?=&A}cMFP(CNl(yMsR`-t2i}7DFz8rYgPveC_)gi
?sXaiv@_U|jtx7a74!l@<;4JHe05F%Q2)SdHLgXxn>Gh!i1WT2K^_-Mqe1LMOvU4R{fH+QfQ%eQYa2d
+e#MFwQ*oaQwvhGC2wTnRW_zJ##J9Pw*x1bE%az6lfqS#7Kz)e-Rnn7GhG_W5G{(q)4xvM*<E{C*%%z
|@dedK@>rJ>eb1rMlG<Q#PU6-U6LF}px>rLDy?OC^}{4osL<Xd-&gathY+6!(-G<KQY^Nf<0_?AJ2h%
baI*&h>lt4f7K8F}Z|`B#E1o*9RQ|@+2V@Bv`<7P)h{}C>a!R4k{Eil{;q0l?<O`2W~qlt?@lZyUjHu
vR&7*JrH75V}PZft@P>#-&mQ~4}M0|c2#<t7A2yM82G?A2D5@m!TT*N3NT=q;Asoc+MTS?=ih&;4fKH
wB*C}hn3lz_KnbzR>OI;L{3Tudz_N!_rY+hTGzghD(#5kNVHprZaZYt##W$uR8%mb^&)N6ivKk8Fogh
BMr8%*?0wS)XN@6p#nApa&Y-w9Ew#Zu@h&NSzS79U`3ykgq8X+X_$OD4A`np9UQ(OZ&?G>pd7)u100h
6&Ehk$^P1yyq9_uwBi4bItZ;{YLK4idID%pU;f7}zm-6NU3Bg;MsQ)C_Xl%pd#APfelK6ZX)4Meva<t
#WouD^v)6>rN3gu`&(XOy?+-ilBrvl6uD3dNNZ)`pUd=i?WZkc;yu4_~oaIcdwK6<&R*Ivfg2cB}&44
buBuR0s5klsH#FHwVF%6r=l3b;v1Sm=o@?fr!Fer2uDL9L&_isoatz297jX@o{A}`XCC6WOfkP0%87K
kvdJZNsFYvO_uCPfDQ#!T$k!x23L!YQl05fIp!Qum#J6eLAwB~uW~Tzhl*@BpO*0iZn3-W@i=nr-W|(
11<!9_wC)@f6`9@{Twzk7R0$51^Jc}CuE!GAA9Xvvbt-);$WrHdL_|lAeLBQZ_CO#8e=M(+dKloNd^C
?t#NN4$3mrXN;--92d@3I`3A83y^SY1a{JrSlpYTf<uyg@8?u9hn^MyFZ_gLeQ<UO`!eB)9Uqb|3x48
z73iihbh>w{K!f#O~7gF*~{#IxcX?lmI`bj@X}m2N^O|Q*fI&p?b#N0ES?Pkc+y}Zj6*0Av01gLhWTd
nOTbhdhE1BKPS3Fg`Z@s&2l@TrvgBPRJC~P2NN1QqdwS}`bs=5XEmp~mF78qqjMEpthH9w@9Bbi4O^<
&W#%iuEa|8!D0@I9@+SrhW~?;jIcMk1?8@}gUVbulb{fRenF5C)Hq<sU(uBiQ1*0^E^hAIe0Dgo9vdT
jSUSyEebQ=m&`3&lwsT_taZ=b~|iRqr6=hHb3s!ZBEizuku9O4FbAG`635jgeYT!aSk9!nV%5DV!m`y
pgT+?87k+yw%=72vpWKHm%I43^$<JP-gMJvgke%q|fRx&y(^Qde)}^Hg2FdU5?0tMb)P*KglnUcUeIa
sA=r#r1j^BZ`5=<A4T}Kb#V2XOX(h6NkuXuu<y{ln7@{>^fLi3aX)oB9={Aw5B1Eya}ud)zI$Kgq)lD
w*#4Ftggw^aT0%+Vu2)HvSC$KTnIVg4EflgOXhv&oh3ZSz1L>6#^bnq(m1-OmeK*vFp=M92_79T`!l}
{9`kKpLiSkSXPgGNTXzC<Mi#wGI%2)E8QZ6lns5f$h{oGgD6pb~sS%_2br2JS3y;ztwFeqDdX60Np7C
{4H-5j|G&bD5*L40y&&58oUwE*8cHr4fVjoD1Jb{N5(5{*jSZY}Z%cvtO+)<AYufl$xy&c4Z?4)+A?0
c+%a(ELlNY48bI{NVe$<pj-eZ4#3ki+j%$Ub>M!m$>YmKH1A`kiHiQ*43y-)7dhX|M$wzbh0!*8wWuO
$+?!aMXV))oSMbZk=4=^~Bzkn$82y9L)OTJZ?WBo<WHOe5LjrV}0-gqo9*@{_X22Q$e1wfAO!OQu2Gq
d0EkxC&m$VDOr%ZU)X{|-`&EC`&Pg%8Zn9OW0l#*U@lWj9krdH$|>qw*oo)B@x+ciJET=1kF<^8cqPG
P!?R*vWNM|ELa#g+A5(FI=e?5HVwlmM86Sq%vDSn84<7Nu4Cy@v^93Go0~&XH@{(?4R;W-+{wnaSX4h
dBLaWBKHJuX_r9tZX^)!C5M>y}CCk2Bg3Q180j_`3MbCnUAON=wR_Mw>=B(2!qdobEOu2v5=yT@shGM
~r3jQ4Lc*S5nc8W7-2G(!r^M~cFZ6m}#W&xX`N#98l}tUwm`Ct`*ss)D%~d2{jaf3BD8RZT}pLU*I=(
}#C|8y|~WONGYELk8FDK9$3V(nS{-09xll!z%G^#&nYYK%@=^l2j(@kWt-j^soOk{KTUk>+MW2)n^^6
(IL-fyvERX*?qG*p`hD|5y$?@0$n)X&O2H<zEKt*HL*`y7(dgS2e<W6b`M8O4cy!}N&|O{PSEhrFs!?
%zk_gXksUd-UHQ;gJuNg|85n^_?)!gYWND7#{zr7A;`8{Z%ssd|;b~QWY2#YaQqg_mY1@oiTQ7==&v-
%=^`ShasOqSbjyd!TrLZ^WNp-d~!|4uMvq)WmMdkv>;^6Ex)SV+1jP-txm+iOw9lzzJAF$`cMda)C%T
GVJkVYGtLqIROwK@kZ{Ay>IU@jDOX%0SdYm|x;oqbm2$vloyp_(hz1t7I43OljOG#o7wOvTf?rAeARa
|#tj9{cl%Yb<zELjUf235p$gGdmob9D4eQtrpNi`bG(`#e=u35Pne6XC}{Xg~>U1ah!CbL`!H33}dzr
htU}qX&Y?3r~m~9(#^Nq?X+Q|_9G!Gbecu}-EupvSfdppnjX=t2xj3q;=s^aZd#RHI3bE@&IndzQQe?
i1`zO-;MmiOM@SbDofi_1tz~EAd#Gh=@ohyX!HEeD{|0MK8X+k#$FHr^$7_}l_w`+ZnbT?no-_f_d2^
gF__@%r^IIH%GBQ!NI72pmqqVd1vE@1Pr*4|@_{B@EF0PV~*Do$#zj*ila-F<df4GJQmTtZLVA!4a=h
Nn2OZ>feF<f3hVA_84=y+KZxXO+Gkz*ReDEu_Ovif+0mK(M&GF$CE_X^#<oYDeJIcRBC`g=;?*kt8Ah
2!YfH*1&m`(}K*=8p%0{=V%6^cPXx&?$FBn<+|4biao99nbt-*K7Fy3cuq7ZbJS?J7{qg$RHFe{9lkS
(e{ts6pOTgz5Hd}UL2c}*5&zHh+4ot{=ZO50|XQR000O88%p+8(ZR!S5DWkSy(j<x761SMaA|NaUukZ
1WpZv|Y%gPBV`ybAaCzNYZI9c=5&nL^VynOiRmi;5aNJgain`7vcHx&GcHj?(1F;fU(%KZsaF-`5MgM
!BnO%}gk-S`*rVT0>vbZm^^FGfkZ#J7flbgK~uVgNF>Y#FaF`LaUF7%-+Dl7KV>@&S?9zU2OZ+>URZm
08I^H`XRZB-mZDJ|^~e)wBFx(RzKvAh|7nx7WpE4{G`@lqRnzbUOQa+zItGP;bDTa~9p6_;}JQPNqll
{?c=cqexYp>wOMvQqd?a(Phwky}+65WS0HZFSa?+{nDh^+sm;N5$kqW|%M-jMb-&VrJWYFY;ULN#F04
%D&c_;;kb)4@Ign6Q{aT8=KTs))4rLN4~GJJ9cF{|Jba5iQjiDJrX0$TIOnOF^e8sbtn^X)T$NFj-8@
{iD(+L$w!^1W||6QX|+Kfkl2FcySN}PQI%LV?h@~meaT}{!YWRZ`NhSX?_PZK;&t-(w{Ko2ub;lU!un
ZJX>5qe<=~GOsoIK!+!4%fY?Ln9d#;VG76M;4bMf#m^kaD;@PQA1r)*v2LSj&^GbPMkK6><66k7}t3G
%k;6qC2p4udo4tT?R?rHN8dg)qrSbuz1WRSnNFs+5(4TFfe%EoKWbTh8VSp>k7KDv@TRHLsjAy~-W$1
1NT<M<!PJ21bfzynZ&H$9wfD)jMTu1VvnIGHHz9m~16^3MtkQO>W?#JpWLXRdK6RW#F?EzNxpE#>lp)
L@KQmY