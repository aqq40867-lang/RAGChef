/**
 * Looks up an element by id and checks at runtime that it has the expected
 * type, so a renamed or retyped element in popup.html fails loudly instead of
 * producing `undefined` property reads later.
 */
export function getElement<T extends HTMLElement>(
  id: string,
  type: new () => T,
): T {
  const element = document.getElementById(id);
  if (!(element instanceof type)) {
    throw new Error(`popup.html: expected #${id} to be a ${type.name}`);
  }
  return element;
}
