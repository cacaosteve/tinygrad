import functools, itertools, math
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp
from tinygrad.dtype import dtypes
from tinygrad.helpers import unwrap

def affine_int_bounds(x:UOp) -> tuple[int, int]|None:
  """Bounds for affine integer expressions, preserving RANGE < end dependencies."""
  def const_int(u:UOp) -> int|None:
    if u.op is Ops.CONST: return int(u.val)
    if u.op is Ops.CAST and u.dtype in dtypes.ints+(dtypes.weakint,) and u.src[0].dtype in dtypes.ints+(dtypes.weakint,) \
       and not u.src[0].overflows(u.dtype): return const_int(u.src[0])
    return None

  def form(u:UOp) -> tuple[dict[UOp, int], int]|None:
    if (c:=const_int(u)) is not None: return {}, c
    if u.op is Ops.CAST and u.dtype in dtypes.ints and u.src[0].dtype in dtypes.ints+(dtypes.weakint,):
      return form(u.src[0]) if not u.src[0].overflows(u.dtype) else None
    if u.op in (Ops.PARAM, Ops.SPECIAL, Ops.RANGE): return {u:1}, 0
    if u.op in (Ops.ADD, Ops.SUB):
      if u.dtype not in dtypes.ints+(dtypes.weakint,) or u.overflows(u.dtype): return None
      if (a:=form(u.src[0])) is None or (b:=form(u.src[1])) is None: return None
      sign = -1 if u.op is Ops.SUB else 1
      coeffs = dict(a[0])
      for atom, coeff in b[0].items(): coeffs[atom] = coeffs.get(atom, 0) + sign * coeff
      return coeffs, a[1] + sign * b[1]
    if u.op is Ops.MUL:
      if u.dtype not in dtypes.ints+(dtypes.weakint,) or u.overflows(u.dtype): return None
      if (c:=const_int(u.src[0])) is not None: other = u.src[1]
      elif (c:=const_int(u.src[1])) is not None: other = u.src[0]
      else: return None
      if (f:=form(other)) is None: return None
      return {atom:c*coeff for atom,coeff in f[0].items()}, c*f[1]
    return None

  def bound(f:tuple[dict[UOp, int], int], upper:bool) -> int|None:
    # Substitute each RANGE with either 0 or end-1 while the expression is still affine.
    # Combining coefficients before taking PARAM extrema preserves correlations such as
    # 2*n-1-RANGE(n)-RANGE(n) == 1 at its lower bound.
    coeffs:dict[UOp, int] = {}
    total = f[1]
    for atom, coeff in f[0].items():
      if atom.op is not Ops.RANGE:
        coeffs[atom] = coeffs.get(atom, 0) + coeff
        continue
      if (coeff > 0) != upper: continue
      if (end:=form(atom.src[0]-1)) is None or any(a.op is Ops.RANGE for a in end[0]): return None
      total += coeff*end[1]
      for a, c in end[0].items(): coeffs[a] = coeffs.get(a, 0) + coeff*c
    for atom, coeff in coeffs.items():
      extreme = atom.vmax if (coeff > 0) == upper else atom.vmin
      if not isinstance(extreme, int): return None
      total += coeff*extreme
    return total

  if (f:=form(x)) is None: return None
  lo, hi = bound(f, False), bound(f, True)
  return None if lo is None or hi is None else (lo, hi)

# NOTE: this cache is only on index UOps
@functools.cache
def fold_divmod_general(d: UOp) -> UOp|None:
  x, y = d.src

  if y.vmin==y.vmax==0: raise ZeroDivisionError(f"{'Division' if d.op is Ops.FLOORDIV else 'Mod'} by zero trying to rewrite {x.alu(d.op, y)}")
  # x//y is constant
  if (xdiv:=x//y).vmin == xdiv.vmax: return x - xdiv.vmin*y if d.op is Ops.FLOORMOD else xdiv.const_like(xdiv.vmin)
  # PARAM // c is irreducible
  if x.op is Ops.PARAM and y.op is Ops.CONST and x.arg.multiple_of % y.val == 0: return d.const_like(0) if d.op is Ops.FLOORMOD else None

  # split uops for the rest of the processing
  x_peeled, const = x.pop_const()
  uops_no_const = list(x_peeled.split_uop(Ops.ADD))

  # ** Constant Denominator Rules **
  # these rules strictly require y to be a scalar constant > 0
  if y.op is Ops.CONST and (c := y.val) > 0:
    # nested_div: (x%(k*c))//c -> (x//c)%k (requires k>0); the mod case is handled by remove_nested_mod below
    if d.op is Ops.FLOORDIV and x.op is Ops.FLOORMOD and (k := x.src[1].divides(c)) is not None and k > 0: return x.src[0] // y % k

    # remove_nested_mod in sum: (a%4 + b)%2 -> (a+b)%2
    if d.op is Ops.FLOORMOD:
      new_xs, changed = [], False
      for u in uops_no_const:
        if u.op is Ops.FLOORMOD and u.src[1].divides(c) is not None:
          u = u.src[0]
          changed = True
        new_xs.append(u)
      if changed: return (UOp.usum(*new_xs) + const) % y

    # Shared decomposition for folding rules
    decomp = [(u.divides(f:=u.const_factor()),f) for u in uops_no_const]
    terms, factors = zip(*decomp)

    # fold_divmod_congruence: fold if a is congruent to an expression whose range is between 0 and c
    # try both signs of the remainder for a lone term (covers a binary numerator that crosses one period)
    # or on an exact f%c == c//2 tie; otherwise pick the smaller to keep the product over terms small
    rem_choices = [(r, r-c) if (r:=f%c)*2 == c or len(terms)==1 else (min(r, r-c, key=abs),) for f in factors]
    for rems in itertools.product(*rem_choices):
      if (rem:=sum(r*v for r,v in zip(rems,terms))+const%c).vmin//c==rem.vmax//c:
        if d.op is Ops.FLOORMOD: return rem - rem.vmin//c*c
        return sum((f-r)//c * v for f,r,v in zip(factors,rems,terms)) + const//c + rem.vmin//c

    # A lone coefficient congruent to one can be canonicalized even when its range crosses
    # several periods: ((1-k*c)*r+b)%c == (r+b)%c. Keeping the large negative coefficient
    # would otherwise make late software division needlessly wide.
    if d.op is Ops.FLOORMOD and len(terms) == 1 and factors[0] != 1 and factors[0]%c == 1 and terms[0].vmin >= 0:
      return (terms[0] + const%c) % y

    # gcd_with_remainder: factor out common gcd from numerator
    if (g:=math.gcd(*factors, c)) > 1:
      new_x = unwrap(x_peeled.divides(g)).simplify() + (const//g)%(c//g)
      if new_x.vmin >= 0:
        if d.op is Ops.FLOORMOD: return new_x % (c//g) * g + const%g
        return new_x // (c//g) + const//c

    # nest_by_factor: x//c -> (x//f)//(c//f), x%c -> (x//f%(c//f))*f + b where b=x%f
    # FLOORDIV identity holds for any sign of x; FLOORMOD reconstruction needs x.vmin>=0
    results = []
    for div in {abs(f) for u, f in zip(uops_no_const, factors) if u.op is not Ops.CONST and 1 < abs(f) < c and (c%f)==0}:
      if (newxs := fold_divmod_general(x//div)) is not None:
        if d.op is Ops.FLOORDIV:
          results.append((len(newxs.backward_slice), newxs // (c // div)))
        elif x.vmin >= 0 and newxs.vmin >= 0:
          b_parts = [f%div*t for f, t in zip(factors, terms) if f%div]
          if const % div: b_parts.append(x.const_like(const % div))
          b = UOp.usum(*b_parts) if b_parts else x.const_like(0)
          if 0 <= b.vmin and b.vmax < div:
            results.append((len((r:=(newxs % x.ufix(c//div))*div + b).backward_slice), r))
    if results: return min(results, key=lambda r: r[0])[1]

  # ** Variable Denominator / Fallback Rules **
  # These rules apply to variables OR constants that failed the checks above.
  # Reconstruct all uops including const for these checks.
  all_uops = list(x.split_uop(Ops.ADD))

  # Dependent ranges retain RANGE < end even when global vmin/vmax loses that relationship.
  if x.vmin >= 0 and y.vmin > 0 and (gap:=affine_int_bounds(y-x)) is not None and gap[0] > 0:
    return x if d.op is Ops.FLOORMOD else x.const_like(0)

  # Mixed-radix identity: (a*f+b) < k*f when a<k and b<f. Candidate f comes from
  # a RANGE-bearing numerator term, so it also works when f and k are symbolic.
  if x.vmin >= 0 and y.vmin > 0:
    candidates = {factor.simplify() for u in all_uops for r in u.split_uop(Ops.MUL)
                  if r.op is Ops.RANGE and (factor:=u.divide_exact(r)) is not None}
    for factor in candidates:
      if (k:=y.divide_exact(factor)) is None: continue
      quotients:list[UOp] = []
      remainders:list[UOp] = []
      for u in all_uops:
        (quotients if (q:=u.divide_exact(factor)) is not None else remainders).append(q if q is not None else u)
      a, b = sum(quotients, x.const_like(0)), sum(remainders, x.const_like(0))
      agap, bgap = affine_int_bounds(k-a), affine_int_bounds(factor-b)
      if a.vmin >= 0 and b.vmin >= 0 and agap is not None and bgap is not None and agap[0] > 0 and bgap[0] > 0:
        return x if d.op is Ops.FLOORMOD else x.const_like(0)

  # Reduce coefficients modulo a symbolic denominator when their difference is an affine constant.
  if d.op is Ops.FLOORMOD and x.vmin >= 0 and y.vmin > 0 and y.op is not Ops.CONST:
    new_uops, changed = [], False
    for u in all_uops:
      nu = u
      for r in u.split_uop(Ops.MUL):
        if r.op is not Ops.RANGE or (factor:=u.divide_exact(r)) is None: continue
        factor = factor.simplify()
        if (delta:=affine_int_bounds(factor-y)) is None or delta[0] != delta[1]: continue
        nu, changed = r*delta[0], True
        break
      new_uops.append(nu)
    if changed: return sum(new_uops, x.const_like(0)) % y

  # divide_by_gcd: x//y -> (x//gcd)//(y//gcd)
  # gcd may not exactly divide symbolic x/y; skip instead of asserting.
  gcd = UOp.gcd(*all_uops, y).simplify()
  if not (gcd.op is Ops.CONST and gcd.val==1):
    if (dx:=x.divide_exact(gcd)) is not None and (dy:=y.divide_exact(gcd)) is not None:
      ret = dx.alu(d.op, dy)
      return ret*gcd if d.op is Ops.FLOORMOD else ret

  # factor_remainder: (d*x+y)//d -> x+y//d
  if y.vmin<0 or x.vmin<0: return None
  quo, rem = [], []
  for u in all_uops:
    if (q:=u.divide_exact(y)) is not None: quo.append(q)
    elif y.op is Ops.CONST and (c:=u.const_factor())%y.val!=c:
      rem.append(u.divides(c)*(c%y.val))
      quo.append(u.divides(c)*(c//y.val) if d.op is Ops.FLOORDIV else u.const_like(0))
    else: rem.append(u)

  if not quo: return None
  new_x = sum(rem)+x.const_like(0)
  if new_x.vmin<0: return None
  return new_x%y if d.op is Ops.FLOORMOD else new_x//y+sum(quo)

div_and_mod_symbolic = PatternMatcher([
  # ** 1. Fast Inline Rules **
  # (x//c+a)//d -> (x+a*c)//(c*d) for c>0, d>0
  ((UPat.var("x")//UPat.cvar("c") + UPat.cvar("a"))//UPat.cvar("d"), lambda x,c,a,d: (x+a*c)//(c*d) if d.vmin>0 else None),
  # (x+c)//d -> (x+c%d)//d + c//d ; (x+c)%d -> (x+c%d)%d  (split the multiple of d out of the const, holds for any d!=0)
  (UPat((Ops.FLOORDIV, Ops.FLOORMOD), src=(UPat.var("x", dtypes.weakint)+UPat.cvar("c"), UPat.cvar("d")), name="n"),
    lambda n,x,c,d: None if d.val==0 or c.val%d.val==c.val else
      (x+c.val%d.val)//d + c.val//d.val if n.op is Ops.FLOORDIV else (x+c.val%d.val)%d),

  # ** 2. Slow Rules **
  (UPat((Ops.FLOORDIV, Ops.FLOORMOD), dtypes.weakint, name="d"), lambda d: fold_divmod_general(d)),
])
