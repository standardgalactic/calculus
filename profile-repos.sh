#!/usr/bin/env bash

JOBS=${JOBS:-$(nproc 2>/dev/null || echo 4)}
DIRTY=${DIRTY:-0}
now=$(date +%s)
export now DIRTY

profile_one() {
  d=$1; repo=${d%/}
  [ -e "$d.git" ] || return

  # date (epoch + short) and commit count
  read -r last_epoch last_date < <(git -C "$d" log -1 --format='%ct %cd' --date=short 2>/dev/null)
  if [ -n "$last_epoch" ]; then
    commits=$(git -C "$d" rev-list --count HEAD 2>/dev/null)
    days=$(( (now - last_epoch) / 86400 ))
    if   [ "$days" -lt 30  ]; then age=fresh
    elif [ "$days" -lt 180 ]; then age=warm
    elif [ "$days" -lt 365 ]; then age=cold
    else age=dormant; fi
  else
    last_date=—; commits=0; age=empty
  fi

  read -r code tex pdf md nb web media data proof test readme tracked sep top_ext < <(
    git -C "$d" ls-files 2>/dev/null | awk '
      function ext(f,  n,a){ n=split(f,a,"."); return (n<2)?"(none)":tolower(a[n]) }
      { t++; base=$0; sub(/.*\//,"",base); e=ext($0); cnt[e]++
        if (e ~ /^(py|c|h|hpp|cpp|hs|rkt|rs|js|ts|go|jl|scm|lisp|el|sh)$/) code++
        else if (e=="ipynb") nb++
        else if (e=="tex") tex++
        else if (e=="pdf") pdf++
        else if (e=="md"||e=="rst") md++
        else if (e=="html"||e=="css"||e=="svg") web++
        else if (e~/^(mp3|wav|m4a|srt|vtt)$/) media++
        else if (e~/^(csv|h5|hdf5|npz|npy|parquet)$/) data++
        else if (e~/^(lean|v|thy)$/) proof++
        if (base ~ /^test_/ || base ~ /_test\./ || base=="Spec.hs" || base ~ /-tests\./ \
            || $0 ~ /(^|\/)validate_/ || $0 ~ /(null_|intervention|permut)/) test++
        if (tolower(base) ~ /^readme/) readme=1 }
      END{ top=""
        for (i=0;i<3;i++){ best=""; bc=-1
          for (k in cnt) if (cnt[k]>bc && !(k in used)){ bc=cnt[k]; best=k }
          if (best=="") break; used[best]=1; top=top best ":" cnt[best] " " }
        sub(/ $/,"",top); if(top=="") top="—"
        printf "%d %d %d %d %d %d %d %d %d %d %d %d | %s\n",
          code+0,tex+0,pdf+0,md+0,nb+0,web+0,media+0,data+0,proof+0,test+0,readme+0,t+0,top }')
  : "$sep"

  sig=""
  [ "${test:-0}"  -gt 0 ] && sig+="tests "
  [ "${proof:-0}" -gt 0 ] && sig+="proofs "
  [ "${data:-0}"  -gt 0 ] && sig+="data "
  [ "${nb:-0}"    -gt 0 ] && sig+="notebooks "
  [ "${readme:-0}" -gt 0 ] && sig+="readme "
  if [ "$DIRTY" = 1 ]; then
    [ -n "$(git -C "$d" status --porcelain -uno 2>/dev/null)" ] && sig+="dirty "
  fi
  sig=${sig% }; [ -z "$sig" ] && sig=—

  doc=$(( ${tex:-0} + ${pdf:-0} + ${md:-0} ))
  nonmedia=$(( ${code:-0} + ${nb:-0} + doc + ${web:-0} ))
  case $repo in
    *backup*|old-*|*-backup|*-old|research-backup) register=archive ;;
    *)
      if   [ "${proof:-0}" -gt 0 ] || [ "${test:-0}" -gt 0 ] || [ "${data:-0}" -gt 0 ]; then register=apparatus
      elif [ "${tracked:-0}" -eq 0 ]; then register=empty
      elif [ "${media:-0}" -gt "$nonmedia" ]; then register=media
      elif [ "${web:-0}" -gt "${code:-0}" ] && [ "${web:-0}" -gt "$doc" ]; then register=viz
      elif [ "$doc" -gt "${code:-0}" ] && [ "$doc" -gt "${nb:-0}" ]; then register=exposition
      elif [ $(( ${code:-0} + ${nb:-0} )) -gt 0 ]; then register=code
      else register=other; fi ;;
  esac

  size_kib=$(git -C "$d" count-objects -v 2>/dev/null | awk '/^size:/{s=$2} /^size-pack:/{p=$2} END{print s+p+0}')
  size=$(awk -v k="${size_kib:-0}" 'BEGIN{ if(k<1024) printf "%dK",k; else if(k<1048576) printf "%.1fM",k/1024; else printf "%.1fG",k/1048576 }')

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$repo" "$register" "$age" "$last_date" "$commits" "${tracked:-0}" "$size" "$sig" "$top_ext"
}
export -f profile_one

printf 'repo\tregister\tage\tlast_commit\tcommits\ttracked\tsize\tsignals\ttop_ext\n'
printf '%s\n' */ | xargs -P "$JOBS" -I{} bash -c 'profile_one "$@"' _ {}
