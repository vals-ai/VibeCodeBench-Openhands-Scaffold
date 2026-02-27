from model_library.base import (
    InputItem,
    ToolResult,
)
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage
from openai.types.responses.response_reasoning_item import (
    ResponseReasoningItem,
)

from openhands.core.logger import openhands_logger as logger


class HistoryWindowCondenser:
    def _group_tool_chains(self, history: list[InputItem]) -> list[tuple[int, ...]]:
        grouped_indices: set[int] = set()
        groups: list[tuple[int, ...]] = []

        # First pass: Find ResponseReasoningItem and check what follows
        for i in range(len(history) - 1):
            if isinstance(history[i], ResponseReasoningItem):
                next_item = history[i + 1]

                if isinstance(next_item, ResponseOutputMessage) or isinstance(
                    next_item, ResponseFunctionToolCall
                ):
                    groups.append((i, i + 1))
                    grouped_indices.add(i)
                    grouped_indices.add(i + 1)

        # Second pass: Add any ungrouped items in, insuring tool call results get merged into the previous group
        for i in range(len(history)):
            if i not in grouped_indices:
                if isinstance(history[i], ToolResult):
                    prev_group = next(
                        (group for group in groups if (i - 1) in group), None
                    )
                    if prev_group:
                        groups[groups.index(prev_group)] = prev_group + (i,)
                        grouped_indices.add(i)
                else:
                    groups.append((i,))

        groups.sort(key=lambda g: g[0])

        return groups

    def _find_essential_items(self, history: list[InputItem]) -> set[int]:
        """Find essential initial items.

        Since InputItems lack role information (unlike typed Events),
        we keep the first few items which typically contain system message,
        initial user message, and initial setup. This mirrors the behavior
        of ConversationWindowCondenser for typed events.

        Args:
            history: List of InputItem objects

        Returns:
            Set of indices containing essential initial items
        """
        # Keep first N items where N = min(5, 25% of history)
        # This captures system message, initial user message, and initial setup
        essential_count = min(5, len(history) // 4)
        return set(range(essential_count))

    def _preserve_tool_call_chains(
        self, history: list[InputItem], items_to_keep: set[int]
    ) -> None:
        """Ensure complete tool call chains are preserved.

        Uses _group_tool_chains to identify complete chains and ensures
        if any item in a chain is kept, all items in that chain are kept.

        Args:
            history: List of InputItem objects
            items_to_keep: Set of indices to keep (modified in-place)
        """
        groups = self._group_tool_chains(history)

        # For each group, if any index is in items_to_keep, add all indices from that group
        for group in groups:
            if any(idx in items_to_keep for idx in group):
                items_to_keep.update(group)
                logger.debug(f'Preserving complete tool chain at indices {group}')

    def condense_history(
        self, history: list[InputItem], preserve_count: int | None = None
    ) -> list[InputItem]:
        """Condense history by keeping essential initial items and the most recent items, while preserving tool call/result chains.

        - Keeps at least 5 items or 25% of history to preserve core context (system prompt, first user message, etc).
        - Keeps roughly half of non-essential recent items, skips dangling ToolResults at the slice boundary, and ensures all tool call chains are kept intact.

        Args:
            history: List of InputItem objects representing conversation history
            preserve_count: Number of recent items to keep (default: half of non-essential)

        Returns:
            Condensed list of InputItem objects maintaining chronological order

        Example:
            >>> history = [
            ...     TextInput(text="System: You are an assistant"),  # 0 - essential (system)
            ...     TextInput(text="User: Help me"),                 # 1 - essential (first user)
            ...     TextInput(text="<REPOSITORY_INFO>..."),          # 2 - essential (workspace)
            ...     TextInput(text="User: Run tests"),               # 3 - middle (forgotten)
            ...     ToolResult(...),                                 # 4 - middle (forgotten)
            ...     TextInput(text="User: Check logs"),              # 5 - recent
            ...     MockToolCall(...),                               # 6 - recent (kept with result)
            ...     ToolResult(...),                                 # 7 - recent
            ... ]
            >>> condenser = HistoryWindowCondenser()
            >>> condensed = condenser.condense_history(history)
            >>> # Result: [0, 1, 2, 5, 6, 7]
            >>> # Items 6 and 7 stay together (tool chain preserved)
        """

        # 1. Don't condense small histories
        if not history or len(history) < 10:
            return history

        # 2. Keep setup items
        essential_indices = self._find_essential_items(history)

        # 3. Find recent items to keep
        num_essential = len(essential_indices)
        total_items = len(history)
        num_non_essential = total_items - num_essential

        # Keep roughly half of non-essential items
        if preserve_count is None:
            preserve_count = max(1, num_non_essential // 2)

        # Calculate starting index for recent section
        slice_start_index = total_items - preserve_count
        # Make sure we don't overlap with essential items
        max_essential_idx = max(essential_indices) if essential_indices else -1
        slice_start_index = max(max_essential_idx + 1, slice_start_index)

        # 4. Find the first valid item in the recent section
        first_valid_index = slice_start_index
        for i in range(slice_start_index, total_items):
            if not isinstance(history[i], ToolResult):
                first_valid_index = i
                break
        else:
            first_valid_index = total_items
            logger.warning('All recent items are dangling ToolResults')

        if first_valid_index > slice_start_index:
            logger.debug(
                f'Skipped {first_valid_index - slice_start_index} dangling ToolResult(s) '
                + 'at the start of recent section'
            )

        # 5. Build items to keep
        items_to_keep: set[int] = set(essential_indices)
        for i in range(first_valid_index, total_items):
            items_to_keep.add(i)

        # 6. Preserve tool call chains to avoid API errors
        self._preserve_tool_call_chains(history, items_to_keep)

        # 7. Build condensed list maintaining order
        condensed = [history[i] for i in sorted(items_to_keep)]

        # Calculate how many extra items were preserved for tool chains
        num_preserved = (
            len(items_to_keep)
            - len(essential_indices)
            - (total_items - first_valid_index)
        )

        logger.info(
            f'HistoryWindowCondenser: {len(history)} → {len(condensed)} items. '
            + f'Essential: {len(essential_indices)}, '
            + f'Recent: {total_items - first_valid_index}, '
            + f'Tool chain preserved: {num_preserved}'
        )

        return condensed
