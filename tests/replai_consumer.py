#!/usr/bin/env python3
"""Observe the production chat process across the installed REPLAI boundary."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import pty
import select
import struct
import subprocess
import termios
import time
import tempfile

ENABLE = b'\x1b[?2004h'
DISABLE = b'\x1b[?2004l'
LABEL = b'deepseek4-v4-flash-dspark'


class Chat:
    def __init__(self, binary, name, output, *, plain=False, dumb=False, memcheck=None, model=None):
        self.master, self.slave = pty.openpty()
        self.name, self.output = name, output
        fcntl.ioctl(self.slave, termios.TIOCSWINSZ, struct.pack('HHHH', 30, 100, 0, 0))
        original = termios.tcgetattr(self.slave)
        original[0] ^= termios.IXOFF
        original[6][termios.VMIN] = 3
        original[6][termios.VTIME] = 7
        termios.tcsetattr(self.slave, termios.TCSANOW, original)
        self.before = termios.tcgetattr(self.slave)
        env = os.environ.copy()
        env.pop('NO_COLOR', None)
        env['TERM'] = 'dumb' if dumb else 'xterm-256color'
        if plain: env['NO_COLOR'] = ''
        command = [str(binary), 'chat', '--session', name]
        if model: command += ['--model', model, '--max-new-tokens', '3']
        self.memlog = output / (name + '.memcheck')
        if memcheck:
            command = [memcheck, '--leak-check=full', '--show-leak-kinds=definite,indirect',
                       '--errors-for-leak-kinds=definite,indirect', '--error-exitcode=99',
                       '--log-file=' + str(self.memlog), *command]
        self.process = subprocess.Popen(command, stdin=self.slave, stdout=self.slave, stderr=self.slave,
            env=env, start_new_session=True, preexec_fn=lambda: fcntl.ioctl(0, termios.TIOCSCTTY, 0))
        self.data = bytearray()
        self.wait(ENABLE)
        self.quiet()
        assert self.before != termios.tcgetattr(self.slave)
        self.fd_baseline = self.tty_fds()
        assert self.fd_baseline == 5, self.fd_baseline  # caller 0/1/2 and two library duplicates

    def pump(self, timeout=0.02):
        if select.select([self.master], [], [], timeout)[0]:
            self.data.extend(os.read(self.master, 65536))

    def quiet(self):
        until = time.monotonic() + 0.15
        while time.monotonic() < until: self.pump()

    def wait(self, needle, start=0):
        deadline = time.monotonic() + 20
        while needle not in self.data[start:]:
            if time.monotonic() > deadline or self.process.poll() is not None:
                raise AssertionError((needle, bytes(self.data[-6000:])))
            self.pump()
        return bytes(self.data[start:])

    def send(self, data):
        if isinstance(data, str): data = data.encode()
        start = len(self.data)
        assert os.write(self.master, data) == len(data)
        return start

    def tty_fds(self):
        target = os.ttyname(self.slave)
        count = 0
        for descriptor in Path(f'/proc/{self.process.pid}/fd').iterdir():
            try: count += os.readlink(descriptor) == target
            except FileNotFoundError: pass
        return count

    def submit(self, data, expected, host_log):
        start = self.send(data + b'\r')
        self.wait(ENABLE, start)
        self.quiet()
        wanted = f'replai-input {self.name} {expected.hex()}\n'
        assert wanted in host_log.read_text(), (wanted, host_log.read_text()[-2000:])
        assert b'hello from yvex' in self.data[start:]
        assert self.tty_fds() == self.fd_baseline
        print(f'chat input={expected.hex()} streamed=hello from yvex next_prompt=1 tty_fds=5', flush=True)

    def finish(self, exit_input=b'\x04'):
        self.send(exit_input)
        deadline = time.monotonic() + 20
        while self.process.poll() is None and time.monotonic() < deadline: self.pump()
        assert self.process.wait(timeout=1) == 0
        self.quiet()
        assert termios.tcgetattr(self.slave) == self.before, 'captured termios not restored'
        assert self.data.count(ENABLE) == self.data.count(DISABLE)
        for forbidden in [b'\x1b[?1049h', b'\x1b[48;', b'\x1b[40m']:
            assert forbidden not in self.data
        if self.memlog.exists():
            memory = self.memlog.read_text()
            assert 'ERROR SUMMARY: 0 errors' in memory, memory
            print('\n'.join(line for line in memory.splitlines() if any(tag in line for tag in
                  ['in use at exit', 'definitely lost', 'indirectly lost', 'ERROR SUMMARY'])), flush=True)
        print(f'{self.name}: termios before==after; paste balanced; exit=0', flush=True)

    def dispose(self):
        (self.output / (self.name + '.typescript')).write_bytes(self.data)
        if self.process.poll() is None:
            self.process.kill(); self.process.wait()
        os.close(self.master); os.close(self.slave)


def dependency_rejections(root, pin):
    # Rejected preparations must neither download/build nor replace an existing prefix.
    with tempfile.TemporaryDirectory(prefix='yvex-terminal-pin-') as temp:
        prefix = Path(temp) / 'prefix'
        prefix.mkdir()
        sentinel = prefix / 'keep'
        sentinel.write_text('caller-owned')
        command = ['python3', str(root / 'tools/prepare_replai.py'), '--prefix', str(prefix)]
        def rejected(expected, arguments=()):
            result = subprocess.run(command + list(arguments), text=True, capture_output=True)
            assert result.returncode != 0 and expected in result.stderr, result
            assert sentinel.read_text() == 'caller-owned'
        rejected('without a verified build receipt')
        rejected('exact pinned revision', ['--source', str(root)])
        receipt = prefix / 'replai-build.json'
        receipt.write_text(json.dumps({'pin': dict(pin, abi=999)}))
        rejected('incompatible REPLAI prefix')
        receipt.write_text(json.dumps({'pin': pin, 'sha256': {'include/replai.h': '0' * 64}}))
        rejected('No such file')
        (prefix / 'include').mkdir()
        (prefix / 'include/replai.h').write_text('incompatible header')
        rejected('artifact integrity failure')
    print('dependency rejection: wrong revision, stale ABI receipt, missing/tampered artifacts; prefix preserved', flush=True)


def audit(binary):
    root = Path(__file__).resolve().parents[1]
    prefix = Path(os.environ['REPLAI_PREFIX'])
    pin = json.loads((root / 'config/replai.json').read_text())
    receipt = json.loads((prefix / 'replai-build.json').read_text())
    assert receipt['pin'] == pin
    dependency_rejections(root, pin)
    symbols = subprocess.check_output(['nm', str(binary)], text=True)
    imports = subprocess.check_output(['nm', '-u', os.environ['YVEX_CLIENT_LANE_OBJ']], text=True)
    for symbol in ['replai_abi_version', 'replai_create', 'replai_open', 'replai_poll',
                   'replai_complete', 'replai_external_output', 'replai_history_add',
                   'replai_interrupt', 'replai_destroy']:
        assert any(line.split()[-1] == symbol for line in symbols.splitlines())
        assert any(line.split()[-1] == symbol for line in imports.splitlines())
    loader = subprocess.check_output(['ldd', str(binary)], text=True)
    assert 'libreplai' not in loader
    source = (root / 'src/cli/io/client.c').read_text()
    for retired in ['repl_read_line(', 'repl_redraw(', 'repl_insert_byte(',
                    'repl_escape_read(', 'repl_columns(', 'repl_erase(']:
        assert retired not in source
    assert '\\033[?2004' not in source
    print(f'product linkage: static REPLAI ABI {pin["abi"]}, revision={pin["revision"]}; old editor absent', flush=True)


def run(binary, host_log, output, memcheck):
    for name, plain, dumb in [('styled', False, False), ('plain', True, False), ('dumb', False, True)]:
        c = Chat(binary, 'replai-' + name, output, plain=plain, dumb=dumb, memcheck=memcheck)
        try:
            prompt = b'\r\x1b[2K' + (LABEL + b'> ' if plain or dumb else b'\x1b[38;5;81m' + LABEL + b'>\x1b[0m ')
            assert ENABLE + prompt in c.data
            if plain or dumb:
                import re
                assert not re.search(rb'\x1b\[[0-9;]*m', c.data)
            c.finish()
        finally: c.dispose()
    c = Chat(binary, 'replai-edit', output, plain=True, memcheck=memcheck)
    try:
        # Combining mark deletion is a grapheme operation; protocol observes exact bytes.
        c.submit('Aé界🌍\x1b[D\x7fX'.encode(), 'AéX🌍'.encode(), host_log)
        c.submit(b'draft\x1b[D\x1b[A\x1b[BX', b'drafXt', host_log)
        c.submit(b'abc\x04\x01\x04', b'bc', host_log)
        c.submit('\x1b[200~é\r\n界\x1b[201~'.encode(), 'é\n界'.encode(), host_log)
        assert b'... ' in c.data
        start = c.send(b'/sta\t')
        c.wait(b'/status', start)
        start = c.send(b'\r'); c.wait(ENABLE, start)
        c.quiet()
        assert b'context' in c.data[start:] and b'hello from yvex' not in c.data[start:]
        start = c.send('wide 界🌍'.encode()); c.wait('界🌍'.encode(), start)
        fcntl.ioctl(c.slave, termios.TIOCSWINSZ, struct.pack('HHHH', 20, 32, 0, 0))
        c.quiet()
        c.send(b'\x0c'); c.wait(b'\x1b[2J\x1b[H', start)
        c.submit(b'\x1b[D!', 'wide 界!🌍'.encode(), host_log)
        # Editing interrupt never sends a generation cancellation request.
        cancel_before = host_log.read_text().count('generation.cancel replai-edit')
        start = c.send(b'unsubmitted\x03'); c.wait(ENABLE, start); c.quiet()
        assert b'^C' in c.data[start:]
        assert host_log.read_text().count('generation.cancel replai-edit') == cancel_before
        # While generation owns output, only the caller's three TTY FDs remain.
        start = c.send(b'WAIT_PREFILL_CANCEL\r')
        c.wait(b'processing 4 input tokens', start)
        assert c.tty_fds() == 3
        flags = termios.tcgetattr(c.slave)[3]
        assert flags & termios.ICANON and flags & termios.ISIG and not flags & termios.ECHO
        c.send(b'\x03'); c.wait(b'cancelled', start); c.wait(ENABLE, start); c.quiet()
        assert 'generation.cancel replai-edit' in host_log.read_text()
        assert c.tty_fds() == 5
        print('generation transition: tty_fds 5->3->5; ICANON/ISIG restored; Ctrl-C routed to cancellation', flush=True)
        for index in range(20):
            text = f'repeat-{index}'.encode()
            c.submit(text, text, host_log)
        c.finish()
    finally: c.dispose()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--host-log', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--memcheck')
    args = parser.parse_args()
    audit(args.binary.resolve())
    run(args.binary.resolve(), args.host_log, args.output, args.memcheck)
