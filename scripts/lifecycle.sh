# lifecycle.sh: the terminal tab owns the servers.
#
# Source it from a start script, call `lifecycle_init "$0" "$@"` before starting anything, start the
# servers in the background, then call `lifecycle_wait <server pids>`. Pressing Ctrl+C, closing the
# terminal tab, the servers being stopped from elsewhere, or the script dying in any other way stops
# every process the script started, helpers included.
#
# How: the script runs as its own process group, and every server and helper it starts joins that
# group. On Ctrl+C or a closed tab (SIGINT, SIGHUP, SIGTERM) the script signals the whole group. A
# small guardian outside the group watches the script; once the script is gone, for whatever
# reason, it gives the group a few seconds to shut down cleanly, kills what is left, and exits.
# Works with the bash 3.2 that ships with macOS.

LIFECYCLE_GRACE_SECONDS="${LIFECYCLE_GRACE_SECONDS:-5}"

lifecycle_init() {
  # From an interactive terminal the shell already gives the script its own process group. Launched
  # from another program we would share that program's group, so start again in a fresh one.
  local pg
  pg=$(ps -o pgid= -p $$ | tr -d ' ')
  if [[ "$pg" != "$$" ]]; then
    if [[ -n "${LIFECYCLE_REEXEC:-}" ]]; then
      echo "lifecycle: could not get an own process group; refusing to start" >&2
      exit 1
    fi
    export LIFECYCLE_REEXEC=1
    exec /usr/bin/perl -e 'setpgrp(0, 0) or die "setpgrp: $!"; exec { $ARGV[0] } @ARGV or die "exec: $!"' /bin/bash "$@"
  fi
  LIFECYCLE_PGID=$$
  trap lifecycle_stop INT TERM HUP
  trap lifecycle_stop EXIT
  lifecycle_guardian
}

lifecycle_guardian() {
  # Detached (own process group, immune to the hang-up signal), so it outlives the script.
  nohup /usr/bin/perl -e '
    setpgrp(0, 0);
    my ($owner, $group, $grace) = @ARGV;
    sleep 1 while kill 0, $owner;            # wait for the start script to end, however it ends
    exit 0 unless kill 0, -$group;            # nothing left: done
    kill "TERM", -$group;
    for (1 .. $grace * 10) {
      last unless kill 0, -$group;
      select(undef, undef, undef, 0.1);
    }
    kill "KILL", -$group if kill 0, -$group;
  ' "$$" "$LIFECYCLE_PGID" "$LIFECYCLE_GRACE_SECONDS" >/dev/null 2>&1 &
  disown $!                                   # keep it out of `wait`, or the two would wait on each other
}

lifecycle_stop() {
  trap - EXIT
  trap '' INT TERM HUP                        # the script itself rides out its own signal
  echo ""
  echo "Stopping all services..."
  kill -TERM -- "-$LIFECYCLE_PGID" 2>/dev/null
  exit 0                                      # the guardian finishes the job
}

lifecycle_wait() {
  # Pass the pids of the server jobs. Helpers that never end on their own (log tails) are left out,
  # so the script also ends, and takes the helpers with it, when the servers are stopped elsewhere.
  echo ""
  echo "Running. Press Ctrl+C or close this terminal tab to stop everything."
  # `wait` returns as soon as a trapped signal arrives, or when the given servers have exited.
  wait "$@"
}
