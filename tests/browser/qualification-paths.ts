import { existsSync, realpathSync } from 'node:fs';
import path from 'node:path';

const canonicalizeNearestExistingAncestor = (candidate: string) => {
  let existing = path.resolve(candidate);
  const missingSegments: string[] = [];

  while (!existsSync(existing)) {
    const parent = path.dirname(existing);
    if (parent === existing) break;
    missingSegments.unshift(path.basename(existing));
    existing = parent;
  }

  return path.resolve(realpathSync.native(existing), ...missingSegments);
};

export const canonicalContainedChild = (root: string, candidate: string) => {
  const canonicalRoot = canonicalizeNearestExistingAncestor(root);
  const canonicalCandidate = canonicalizeNearestExistingAncestor(candidate);
  const relative = path.relative(canonicalRoot, canonicalCandidate);
  return relative !== '' && !relative.startsWith('..') && !path.isAbsolute(relative);
};

export function resolveQualificationCapture(checkoutRoot: string, requested: string | undefined) {
  if (!requested) throw new Error('qualification capture path is required');
  const root = path.resolve(
    checkoutRoot,
    'artifacts',
    'diagnostics',
    'variable-duration',
  );
  const capture = path.resolve(checkoutRoot, requested);
  if (!canonicalContainedChild(root, capture))
    throw new Error('qualification capture must be a contained nonempty child');
  return capture;
}

export function assertFreshQualificationCapture(
  capture: string,
  exists: (candidate: string) => boolean,
) {
  if (exists(capture)) throw new Error(`qualification capture already exists: ${capture}`);
}

export function resolveQualificationProfile(
  checkoutRoot: string,
  requested: string | undefined,
  linkedWorktree: boolean,
) {
  const profileRoot = path.resolve(
    checkoutRoot,
    'artifacts',
    linkedWorktree ? 'worktree-profiles' : 'browser-profiles',
  );
  if (linkedWorktree && !requested)
    throw new Error('linked worktrees require an explicit Chrome profile');
  const profile = path.resolve(
    checkoutRoot,
    requested ?? 'artifacts/browser-profiles/variable-duration/task11',
  );
  if (!canonicalContainedChild(profileRoot, profile)) {
    const rootName = linkedWorktree ? 'worktree-profiles' : 'browser-profiles';
    throw new Error(`Chrome profile must be a nonempty child of artifacts/${rootName}`);
  }
  return profile;
}
