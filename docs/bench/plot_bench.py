import json
import math
import statistics

# --- load data: group rows by cell count, collect all runs per GPU ---
runs = {}
with open("gpus.txt") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=3)
        size = int(float(parts[0]))
        runs.setdefault(size, {}).setdefault(parts[3], []).append(float(parts[2]) / 1e6)

# mean, standard deviation and number of runs per GPU
# (a single run has no spread, so its standard deviation is 0 and no error bar is drawn)
groups = {}
for size, gpus in runs.items():
    groups[size] = {
        lab: [statistics.mean(v),
              statistics.stdev(v) if len(v) > 1 else 0.0,
              len(v)]
        for lab, v in gpus.items()
    }

sizes = sorted(groups)

# --- layout (SVG units; the SVG scales to the page width) ---
W, H = 800, 400
left, right, top = 70, 20, 45
longest = max(len(l) for g in groups.values() for l in g)
bottom = int(longest * 6.3) + 20                  # same label space for every group

data = {"sizes": sizes, "vals": {str(s): groups[s] for s in sizes}}
data_js = json.dumps(data).replace("</", "<\\/")

template = """<div id="gpubench" style="position:relative;max-width:800px;margin:auto;font-family:sans-serif">
<style>
#gpubench .bar{fill:#1f77b4}
#gpubench .bar:hover{fill:#ff7f0e}
#gpubench .tip{position:absolute;pointer-events:none;display:none;padding:4px 8px;
  background:rgba(30,30,30,.9);color:#fff;border-radius:4px;font-size:12px;white-space:nowrap}
#gpubench .ctl{display:flex;align-items:center;gap:12px;justify-content:center;font-size:14px;margin-top:4px}
#gpubench input[type=range]{flex:1;max-width:400px}
</style>
<svg id="gpubench-svg" viewBox="0 0 %W% %H%" width="100%" role="img"></svg>
<div class="ctl">
  <span>Number of cells:</span>
  <input id="gpubench-slider" type="range" min="0" max="%MAX%" step="1" value="9">
  <span id="gpubench-size" style="min-width:9em"></span>
</div>
<div class="ctl">
  <label><input id="gpubench-keep" type="checkbox">
  keep current (GPU) order</label>
</div>
<div class="tip"></div>
<script>
(function(){
  var D=%DATA%;
  var W=%W%, H=%H%, L=%LEFT%, R=%RIGHT%, T=%TOP%, B=%BOTTOM%;
  var PW=W-L-R, PH=H-T-B;
  var sizes=D.sizes, vals=D.vals;      // vals[size][gpu] = [mean, std, runs]

  var box=document.getElementById("gpubench"), tip=box.querySelector(".tip");
  var svg=document.getElementById("gpubench-svg");
  var slider=document.getElementById("gpubench-slider");
  var label=document.getElementById("gpubench-size");
  var keep=document.getElementById("gpubench-keep");
  var frozen=null;

  var SUP={"0":"\\u2070","1":"\\u00b9","2":"\\u00b2","3":"\\u00b3","4":"\\u2074",
           "5":"\\u2075","6":"\\u2076","7":"\\u2077","8":"\\u2078","9":"\\u2079"};

  function sup(e){
    var s="", d=String(e);
    for(var k=0; k<d.length; k++) s+=SUP[d.charAt(k)];
    return s;
  }

  // n = 2^e. For even e the grid is square: 2^(e/2) x 2^(e/2) x 1.
  // Odd e shows as 2^e; other numbers as 1,234,567.
  function cells(n){
    var e=Math.round(Math.log(n)/Math.LN2);
    if(!(n>=1) || Math.pow(2,e)!==n) return n.toLocaleString("en-US");
    if(e%2===0){
      var p="2"+sup(e/2);
      return p+" \\u00d7 "+p+" \\u00d7 1";
    }
    return "2"+sup(e);
  }

  function esc(s){
    return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;")
                    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
  }
  function group(i){ return vals[String(sizes[i])]; }

  // labels of one group, sorted low to high by mean
  function sortedLabels(i){
    var g=group(i);
    return Object.keys(g).sort(function(a,b){return g[a][0]-g[b][0];});
  }

  // order of group i, followed by every other GPU (sorted by its mean in
  // the largest group where it appears) so that missing GPUs keep a slot
  function buildFrozen(i){
    var cur=sortedLabels(i), seen={}, ref={};
    cur.forEach(function(l){seen[l]=true;});
    sizes.forEach(function(s){
      var g=vals[String(s)];
      for(var l in g){ ref[l]=g[l][0]; }
    });
    var rest=Object.keys(ref).filter(function(l){return !seen[l];})
                   .sort(function(a,b){return ref[a]-ref[b];});
    return cur.concat(rest);
  }

  function niceMax(m){
    var raw=m/5, mag=Math.pow(10,Math.floor(Math.log(raw)/Math.LN10)), step=mag;
    [1,2,5,10].some(function(k){ step=k*mag; return step>=raw; });
    return {ymax:Math.ceil(m/step)*step, step:step};
  }

  function draw(){
    var i=+slider.value, g=group(i);
    var labels=frozen?frozen:sortedLabels(i);
    var n=labels.length;
    // axis maximum includes the top of the error bars
    var m=0; labels.forEach(function(l){
      if(g[l]!==undefined) m=Math.max(m,g[l][0]+g[l][1]);
    });
    var a=niceMax(m), ymax=a.ymax, step=a.step;
    function ypix(v){ return T+PH-v/ymax*PH; }

    var o=[];
    for(var k=0; k*step<=ymax*(1+1e-9); k++){
      var v=k*step, y=ypix(v);
      o.push('<line x1="'+L+'" x2="'+(L+PW)+'" y1="'+y.toFixed(1)+'" y2="'+y.toFixed(1)+
             '" stroke="currentColor" stroke-opacity="0.15"/>');
      o.push('<text x="'+(L-8)+'" y="'+(y+4).toFixed(1)+'" text-anchor="end" font-size="12" '+
             'fill="currentColor">'+(+v.toFixed(6))+'</text>');
    }
    o.push('<line x1="'+L+'" x2="'+L+'" y1="'+T+'" y2="'+(T+PH)+'" stroke="currentColor"/>');
    o.push('<line x1="'+L+'" x2="'+(L+PW)+'" y1="'+(T+PH)+'" y2="'+(T+PH)+'" stroke="currentColor"/>');

    var slot=PW/n, bw=0.6*slot;
    labels.forEach(function(lab,j){
      var cx=L+(j+0.5)*slot, e=g[lab], has=(e!==undefined);
      if(has){
        var v=e[0], sd=e[1], cnt=e[2];
        var html=esc(lab)+"<br><b>"+v.toFixed(2)+"</b>"+
                 (cnt>1?" \\u00b1 "+sd.toFixed(2):"")+" M cells/s<br>"+
                 cnt+(cnt===1?" run":" runs");
        o.push('<rect class="bar" x="'+(cx-bw/2).toFixed(1)+'" y="'+ypix(v).toFixed(1)+
               '" width="'+bw.toFixed(1)+'" height="'+(T+PH-ypix(v)).toFixed(1)+
               '" data-tip="'+esc(html)+'"/>');
        if(cnt>1 && sd>0){
          var yt=ypix(v+sd).toFixed(1), yb=ypix(Math.max(v-sd,0)).toFixed(1);
          var c=(bw*0.25).toFixed(1), x0=(cx-c).toFixed(1), x1=(+cx+ +c).toFixed(1);
          o.push('<path d="M'+cx.toFixed(1)+' '+yt+'V'+yb+
                 'M'+x0+' '+yt+'H'+x1+'M'+x0+' '+yb+'H'+x1+
                 '" stroke="currentColor" stroke-width="1.5" fill="none" pointer-events="none"/>');
        }
      }
      var ty=T+PH+8;
      o.push('<text x="'+cx.toFixed(1)+'" y="'+ty+'" font-size="11" fill="currentColor" '+
             'fill-opacity="'+(has?1:0.35)+'" text-anchor="end" '+
             'transform="rotate(-90 '+cx.toFixed(1)+' '+ty+')" dy="4">'+esc(lab)+'</text>');
    });

    o.push('<text x="'+(L+PW/2)+'" y="24" text-anchor="middle" font-size="14" fill="currentColor">'+
           'mumax\\u207a GPU performance for 2D simulations containing '+
           sizes[i].toLocaleString("en-US")+' cells.</text>');
    o.push('<text transform="translate(16 '+(T+PH/2)+') rotate(-90)" text-anchor="middle" '+
           'font-size="12" fill="currentColor">throughput (M cells/s)</text>');

    svg.innerHTML=o.join("\\n");
    label.textContent=cells(sizes[i]);
    tip.style.display="none";
  }

  slider.addEventListener("input",draw);
  keep.addEventListener("change",function(){
    frozen=keep.checked?buildFrozen(+slider.value):null;
    draw();
  });

  // tooltip via event delegation, since the bars are redrawn
  svg.addEventListener("mousemove",function(e){
    var t=e.target;
    if(t.classList && t.classList.contains("bar")){
      var r=box.getBoundingClientRect();
      tip.innerHTML=t.getAttribute("data-tip");
      tip.style.display="block";
      tip.style.left=(e.clientX-r.left+12)+"px";
      tip.style.top=(e.clientY-r.top-30)+"px";
    } else {
      tip.style.display="none";
    }
  });
  svg.addEventListener("mouseleave",function(){tip.style.display="none";});

  draw();
})();
</script>
</div>
"""

out = (template.replace("%W%", str(W)).replace("%H%", str(H))
               .replace("%LEFT%", str(left)).replace("%RIGHT%", str(right))
               .replace("%TOP%", str(top)).replace("%BOTTOM%", str(bottom))
               .replace("%MAX%", str(len(sizes) - 1))
               .replace("%DATA%", data_js))      # data last, so labels can't be altered

with open("../_static/bench.html", "w", encoding="utf-8") as f:
    f.write(out)
